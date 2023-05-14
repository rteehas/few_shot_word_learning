import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaForMaskedLM, AutoModelForCausalLM
from memory import OnlineProtoNet
from utils import *


class MorphMemoryModel(nn.Module):

    def __init__(self, firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config):
        super(MorphMemoryModel, self).__init__()

        self.layers = layers
        self.device = device
        self.mask_token_id = mask_token_id
        self.firstLM = firstLM.to(device)
        self.secondLM = secondLM.to(device)

        self.freeze_roberta()

        self.memory = OnlineProtoNet(memory_config, self.device)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.firstLM.config.hidden_size, nhead=1).to(self.device)
        self.emb_gen = nn.TransformerEncoder(encoder_layer, num_layers=1).to(self.device)

        self.nonces = nonces  # for referencing embedding location

        self.model_name = "memory_model_{}_{}_{}_memory".format(self.firstLM.config.model_type,
                                                                self.secondLM.config.model_type,
                                                                self.memory_config.agg_method)

        # initialize with mean embedding
        indices = self.get_zero_grads_idx()
        m = torch.mean(self.secondLM.roberta.embeddings.word_embeddings.weight[indices, :], dim=0)
        for nonce in nonces:
            with torch.no_grad():
                self.secondLM.roberta.embeddings.word_embeddings.weight[nonce, :] = m
                self.firstLM.roberta.embeddings.word_embeddings.weight[nonce, :] = m

    def freeze_roberta(self, tune_tok=False):
        for parameter in self.firstLM.parameters():
            parameter.requires_grad = False

        if tune_tok:
            self.firstLM.roberta.embeddings.word_embeddings.weight.requires_grad = True

        if self.secondLM.config.model_type == "roberta":  # check if secondLM is roberta
            for parameter in self.secondLM.parameters():
                parameter.requires_grad = False

            if tune_tok:
                self.secondLM.roberta.embeddings.word_embeddings.weight.requires_grad = True

    def forward(self, batch):

        mlm_inputs = batch["mlm_inputs"].to(self.device)
        task_inputs = batch["task_inputs"].to(self.device)
        nonceMLM = batch["nonceMLM"]
        nonceTask = batch["nonceTask"]
        if 'task_labels' in batch:
            task_labels = batch["task_labels"].to(device)
        else:
            task_labels = None

        b, k, l = batch["mlm_inputs"]["input_ids"].shape  # batch x k examples x max_length toks

        mlm_inputs["input_ids"] = self.swap_with_mask(mlm_inputs["input_ids"], k, nonceMLM)  # replace nonce with mask

        mlm_ids = mlm_inputs["input_ids"].reshape((b * k, l))  # reshape so we have n x seq_len
        mlm_attn = mlm_inputs["attention_mask"].reshape((b * k, l))

        first_out = self.firstLM(input_ids=mlm_ids, attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        chunked = torch.chunk(combined, b)  # get different embeds per nonce, shape = k x max_len x hidden

        # embedding generator + store in memory
        for i, chunk in enumerate(chunked):
            msk = (mlm_inputs['input_ids'][i] == self.mask_token_id)  # mask all but the outputs for the mask

            generated_embeds = self.emb_gen(chunk.permute(1, 0, 2), src_key_padding_mask=msk).permute(1, 0,
                                                                                                      2)  # permute for shape, mask, permute back

            embed_ids = msk.nonzero(as_tuple=True)

            nonce_embeds = generated_embeds[embed_ids[0], embed_ids[1], :]

            self.memory.store(nonceMLM[i].item(), nonce_embeds)

        # now do task specific stuff
        b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape

        task_ids = task_inputs["input_ids"].reshape((b_task * k_task, l_task))  # reshape so we have n x seq_len
        task_attn = task_inputs["attention_mask"].reshape((b_task * k_task, l_task))

        if 'task_labels' in batch:
            task_labels = task_labels.reshape((b_task * k_task, l_task))

        if self.secondLM.config.model_type == "roberta":
            w = self.secondLM.roberta.embeddings.word_embeddings.weight.clone()

        weight_mask = torch.nn.functional.one_hot(torch.LongTensor(nonceTask),
                                                  num_classes=self.secondLM.config.vocab_size).to(self.device).sum(
            0).unsqueeze(0)  # get one hot for nonces

        new_w = w * (1 - weight_mask).T

        for nonce in nonceMLM:
            nonce_embed = self.memory.retrieve(nonce.item())

            new_w = new_w + (nonce_embed * weight_mask.T)

        input_embeds = torch.nn.functional.embedding(task_ids, new_w)

        second_out = self.secondLM(inputs_embeds=input_embeds, attention_mask=task_attn, labels=task_labels,
                                   output_hidden_states=True)
        return second_out

    def eval_batch(self, batch):
        self.forward(batch)
        return self.eval_step(batch)

    def eval_step(self, batch):
        probe_inputs = batch['probe_inputs']
        eval_nonce = batch["eval_nonce"].to(self.device)
        nonceMLM = batch['nonceMLM']
        nonceTask = batch['nonceTask']
        ratings = batch['ratings']

        if self.secondLM.config.model_type == "roberta":
            w = self.secondLM.roberta.embeddings.word_embeddings.weight.clone()

        weight_mask = torch.nn.functional.one_hot(torch.LongTensor(nonceTask),
                                                  num_classes=self.secondLM.config.vocab_size).to(self.device).sum(
            0).unsqueeze(0)  # get one hot for nonces

        new_w = w * (1 - weight_mask).T

        for nonce in nonceMLM:
            nonce_embed = self.memory.retrieve(nonce.item())

            new_w = new_w + (nonce_embed * weight_mask.T)

        input_embeds = torch.nn.functional.embedding(eval_nonce['input_ids'].squeeze(0), new_w)
        attn = eval_nonce['attention_mask'].squeeze(0)
        nonce_out = self.secondLM(inputs_embeds=input_embeds, attention_mask=attn, output_hidden_states=True)
        nonce_embed = nonce_out.hidden_states[-1].sum(1)

        probe_embeds = []
        for probe in probe_inputs:
            probe_task = probe_inputs[probe]['task'].to(self.device)
            probe_out = self.secondLM(input_ids=probe_task['input_ids'].squeeze(0),
                                      attention_mask=probe_task['attention_mask'].squeeze(0), output_hidden_states=True)
            probe_embed = probe_out.hidden_states[-1].sum(1)
            probe_embeds.append(probe_embed)
        return nonce_embed, probe_embeds

    def swap_with_mask(self, inputs, k_examples, nonces):
        exp = torch.Tensor(nonces).unsqueeze(1).expand_as(inputs[:, 0, :]).to(
            self.device)  # expand for a set of sentences across batches
        exp = exp.unsqueeze(1).repeat(1, k_examples, 1)  # repeat for k sentences
        inputs[inputs == exp] = self.mask_token_id
        return inputs

    def get_zero_grads_idx(self):
        index_grads_to_zero = torch.empty(self.firstLM.config.vocab_size, dtype=torch.bool).fill_(True)
        for nonce in self.nonces:
            index_grads_to_zero = index_grads_to_zero & (torch.arange(self.firstLM.config.vocab_size) != nonce)
        return index_grads_to_zero

    def initialize_with_def(self, nonce, definition):
        with torch.no_grad():
            toks = self.tokenizer(definition, return_tensors="pt").to(device)
            outs = self.firstLM(**toks, output_hidden_states=True)

    def get_new_weights(self, batch, task="MLM"):

        key = "nonce{}".format(task)

        nonces = batch[key]

        nonceMLM = batch['nonceMLM']

        if task == 'MLM':
            ref_model = self.firstLM

        elif task == "Task":
            ref_model = self.secondLM

        if ref_model.config.model_type == "roberta":
            w = ref_model.roberta.embeddings.word_embeddings.weight.clone()

        weight_mask = torch.nn.functional.one_hot(torch.LongTensor(nonces),
                                                  num_classes=ref_model.config.vocab_size).to(self.device).sum(
            0).unsqueeze(0)  # get one hot for nonces

        new_w = w * (1 - weight_mask).T

        for nonce in nonceMLM:
            nonce_embed = self.memory.retrieve(nonce.item())

            new_w = new_w + (nonce_embed * weight_mask.T)

        return new_w

    def re_encode(self, input_ids, w):

        return torch.nn.functional.embedding(input_ids.squeeze(0), w)


