import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import MaskedLMOutput

from modules.memory import OnlineProtoNet
from modules.utils import combine_layers
from modules.embedding_generators import MLP
from transformers.activations import gelu

class MorphMemoryModel(nn.Module):

    def __init__(self, firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type="MLP"):
        super(MorphMemoryModel, self).__init__()

        self.layers = layers
        self.device = device
        self.mask_token_id = mask_token_id
        self.firstLM = firstLM.to(device)
        self.secondLM = secondLM.to(device)

        self.emb_decoder = nn.Linear(self.firstLM.config.hidden_size, self.firstLM.config.vocab_size).to(device)

        self.freeze_roberta()

        self.memory = OnlineProtoNet(memory_config, self.device)
        self.emb_type = emb_type

        if self.emb_type == "MLP":
            self.emb_gen = MLP(self.firstLM.config.hidden_size, 384, self.secondLM.config.hidden_size)

        elif self.emb_type == "Transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.firstLM.config.hidden_size, nhead=1).to(self.device)
            self.emb_gen = nn.TransformerEncoder(encoder_layer, num_layers=1).to(self.device)



        self.nonces = nonces  # for referencing embedding location

        self.model_name = "memory_model_{}_{}_{}_memory".format(self.firstLM.config.model_type,
                                                                self.secondLM.config.model_type,
                                                                memory_config.agg_method)
    def freeze_roberta(self, tune_tok=False):
        for parameter in self.firstLM.parameters():
            parameter.requires_grad = False

        #         if tune_tok:
        #             self.firstLM.roberta.embeddings.word_embeddings.weight.requires_grad = True

        if self.secondLM.config.model_type == "roberta":  # check if secondLM is roberta
            for parameter in self.secondLM.parameters():
                parameter.requires_grad = False

            if tune_tok:
                self.secondLM.roberta.embeddings.word_embeddings.weight.requires_grad = True

    #         self.secondLM.lm_head.bias.requires_grad=True
    #         self.secondLM.roberta.embeddings.word_embeddings.weight.requires_grad = True

    def forward(self, batch):

        mlm_inputs = batch["mlm_inputs"].to(self.device)
        task_inputs = {'input_ids': batch["task_inputs"]['input_ids'].to(self.device),
                       'attention_mask': batch["task_inputs"]['attention_mask'].to(self.device)}
        nonceMLM = batch["nonceMLM"]
        nonceTask = batch["nonceTask"]
        if 'task_labels' in batch:
            task_labels = batch["task_labels"].to(self.device)

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
        losses = []
        for i, chunk in enumerate(chunked):
            msk = (mlm_inputs['input_ids'][i] == self.mask_token_id)  # mask all but the outputs for the mask

            if self.emb_type == "Transformer":
                generated_embeds = self.emb_gen(chunk.permute(1, 0, 2), src_key_padding_mask=msk).permute(1, 0,
                                                                                                    2)  # permute for shape, mask, permute back

            embed_ids = msk.nonzero(as_tuple=True)

            if self.emb_type == "Transformer":
                nonce_embeds = generated_embeds[embed_ids[0], embed_ids[1], :]

            elif self.emb_type == "MLP":
                nonce_embeds = self.emb_gen(chunk[embed_ids[0], embed_ids[1], :])

            preds = self.emb_decoder(nonce_embeds)
            labs = nonceMLM[i].to(self.device).repeat(preds.shape[0])
            #                 print(preds.shape, nonce_embeds.shape)
            l_fct = nn.CrossEntropyLoss()
            inter_loss = l_fct(preds.view(-1, self.firstLM.config.vocab_size), labs.view(-1))
            losses.append(inter_loss)

            self.memory.store(nonceMLM[i].item(), nonce_embeds)

        # now do task specific stuff
        b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape

        task_ids = task_inputs["input_ids"].reshape((b_task * k_task, l_task))  # reshape so we have n x seq_len
        task_attn = task_inputs["attention_mask"].reshape((b_task * k_task, l_task))

        if 'task_labels' in batch:
            task_labels = task_labels.reshape((b_task * k_task, l_task))

        new_w = self.get_new_weights(task="Task")
        input_embeds = F.embedding(task_ids, new_w)

        #         out_vals = self.secondLM(inputs_embeds=input_embeds, attention_mask=task_attn, labels=task_labels,
        #                                    output_hidden_states=True)

        #         l2 = out_vals.loss
        outputs = self.secondLM.roberta(
            inputs_embeds=input_embeds,
            attention_mask=task_attn,
            output_hidden_states=True
        )

        preds = self.calc_second_lmhead(new_w, outputs[0])
        loss_fct = nn.CrossEntropyLoss()
        lm_loss = loss_fct(preds.view(-1, self.secondLM.config.vocab_size), task_labels.view(-1))
        # #         l = nn.CrossEntropyLoss(reduction="none")

        out_vals = MaskedLMOutput(
            loss=lm_loss,
            logits=preds,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
        return out_vals, losses

    def calc_second_lmhead(self, new_w, last_hidden):
        x = self.secondLM.lm_head.dense(last_hidden)
        x = gelu(x)
        x = self.secondLM.lm_head.layer_norm(x)
        x = F.linear(x, new_w, bias=self.secondLM.lm_head.bias)
        return x

    def eval_batch(self, batch):
        self.forward(batch)
        return self.eval_step(batch)

    def eval_step(self, batch):
        probe_inputs = batch['probe_inputs']
        eval_nonce = batch["eval_nonce"].to(self.device)
        nonceMLM = batch['nonceMLM']
        nonceTask = batch['nonceTask']
        ratings = batch['ratings']

        new_w = self.get_new_weights("Task")

        input_embeds = torch.nn.functional.embedding(eval_nonce['input_ids'].squeeze(0), new_w)
        attn = eval_nonce['attention_mask'].squeeze(0)
        nonce_out = self.secondLM.roberta(
            inputs_embeds=input_embeds,
            attention_mask=attn,
            output_hidden_states=True
        )
        nonce_embed = nonce_out.hidden_states[-1].sum(1)

        probe_embeds = []
        for probe in probe_inputs:
            probe_task = probe_inputs[probe]['task'].to(self.device)
            probe_input_embeds = F.embedding(probe_task['input_ids'].squeeze(0), new_w)
            probe_out = self.secondLM.roberta(inputs_embeds = probe_input_embeds,
                                      attention_mask=probe_task['attention_mask'].squeeze(0),
                                      output_hidden_states=True)
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
            toks = self.tokenizer(definition, return_tensors="pt").to(self.device)
            outs = self.firstLM(**toks, output_hidden_states=True)

    def get_new_weights(self, task="MLM"):

        if task == 'MLM':
            ref_model = self.firstLM

        elif task == "Task":
            ref_model = self.secondLM

        if ref_model.config.model_type == "roberta":
            w = ref_model.roberta.embeddings.word_embeddings.weight.clone()

        a = []
        for i in range(w.shape[0]):
            if i in self.memory.memory:
                a.append(self.memory.retrieve(i))
            else:
                a.append(w[i, :].unsqueeze(0))
        return torch.cat(a, dim=0)

    def update_weights(self):
        first_wt = nn.Embedding(self.firstLM.config.vocab_size,
                                self.firstLM.config.hidden_size).from_pretrained(self.get_new_weights(task='MLM'))
        second_wt = nn.Embedding(self.secondLM.config.vocab_size,
                                 self.secondLM.config.hidden_size).from_pretrained(self.get_new_weights(task='Task'))
        with torch.no_grad():
            self.firstLM.set_input_embeddings(first_wt)

            self.secondLM.set_input_embeddings(second_wt)
        self.freeze_roberta()

    def re_encode(self, input_ids, w):

        return torch.nn.functional.embedding(input_ids.squeeze(0), w)
