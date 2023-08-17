import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_outputs import MaskedLMOutput, QuestionAnsweringModelOutput, \
    CausalLMOutputWithCrossAttentions
import math
from modules.memory import OnlineProtoNet
from modules.utils import combine_layers
from modules.embedding_generators import MLP
from transformers.activations import gelu


class MorphMemoryModel(nn.Module):

    def __init__(self, firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type):
        super(MorphMemoryModel, self).__init__()

        self.layers = layers
        self.device = device
        self.mask_token_id = mask_token_id
        self.firstLM = firstLM.to(device)
        self.secondLM = secondLM.to(device)

        # self.emb_decoder = nn.Linear(self.firstLM.config.hidden_size, self.firstLM.config.vocab_size).to(device)

        self.freeze()

        self.memory = OnlineProtoNet(memory_config, self.device)
        self.emb_type = emb_type

        if self.emb_type == "MLP":
            self.emb_gen = MLP(self.firstLM.config.hidden_size, 384, self.secondLM.config.hidden_size)

        elif self.emb_type == "Transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.firstLM.config.hidden_size,
                                                       nhead=self.firstLM.config.num_attention_heads, activation='gelu').to(self.device)
            self.emb_gen = nn.TransformerEncoder(encoder_layer, num_layers=1).to(self.device)

        self.nonces = nonces  # for referencing embedding location

        # new tokens always at the end
        initial_first_ind = int(self.firstLM.config.vocab_size - len(self.nonces))
        initial_second_ind = int(self.secondLM.config.vocab_size - len(self.nonces))
        m_first = torch.mean(self.firstLM.get_input_embeddings().weight[:initial_first_ind, :], dim=0)
        m_second = torch.mean(self.secondLM.get_input_embeddings().weight[:initial_second_ind, :], dim=0)

        first_list = list(range(initial_first_ind, self.firstLM.config.vocab_size))
        second_list = list(range(initial_first_ind, self.secondLM.config.vocab_size))
        for n_first, n_second in zip(first_list, second_list):
            with torch.no_grad():
                self.firstLM.get_input_embeddings().weight[n_first, :] = m_first
                self.secondLM.get_input_embeddings().weight[n_second, :] = m_second

        self.model_name = "memory_model_{}_{}_{}_memory".format(self.firstLM.config.model_type,
                                                                self.secondLM.config.model_type,
                                                                memory_config.agg_method)

    def freeze(self):
        for parameter in self.firstLM.parameters():
            parameter.requires_grad = False

        for parameter in self.secondLM.parameters():
            parameter.requires_grad = False

    def forward(self, batch):

        mlm_inputs = batch["mlm_inputs"].to(self.device)
        task_inputs = {'input_ids': batch["task_inputs"]['input_ids'].to(self.device),
                       'attention_mask': batch["task_inputs"]['attention_mask'].to(self.device)}
        nonceMLM = batch["nonceMLM"]
        nonceTask = batch['nonceTask']
        if 'task_labels' in batch:
            task_labels = batch["task_labels"].to(self.device)

        else:
            task_labels = None

        b, k, l = batch["mlm_inputs"]["input_ids"].shape  # batch x k examples x max_length toks

        new_inputs = self.swap_with_mask(mlm_inputs["input_ids"], k, nonceMLM)  # replace nonce with mask

        mlm_ids = new_inputs.reshape((b * k, l))  # reshape so we have n x seq_len
        mlm_attn = mlm_inputs["attention_mask"].reshape((b * k, l))

        first_out = self.firstLM(input_ids=mlm_ids, attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        chunked = torch.chunk(combined, b)  # get different embeds per nonce, shape = k x max_len x hidden

        # embedding generator + store in memory
        losses = []
        for i, chunk in enumerate(chunked):
            msk = (new_inputs[i] == self.mask_token_id)  # mask all but the outputs for the mask
            if self.emb_type == "Transformer":
                generated_embeds = self.emb_gen(chunk.permute(1, 0, 2), src_key_padding_mask=~msk).permute(1, 0,
                                                                                                           2)  # permute for shape, mask, permute back

            embed_ids = msk.nonzero(as_tuple=True)

            if self.emb_type == "Transformer":
                nonce_embeds = generated_embeds[embed_ids[0], embed_ids[1], :]

            elif self.emb_type == "MLP":
                nonce_embeds = self.emb_gen(chunk[embed_ids[0], embed_ids[1], :])

            # preds = self.emb_decoder(nonce_embeds)
            # labs = nonceMLM[i].to(self.device).repeat(preds.shape[0])
            # #                 print(preds.shape, nonce_embeds.shape)
            # l_fct = nn.CrossEntropyLoss()
            # inter_loss = l_fct(preds.view(-1, self.firstLM.config.vocab_size), labs.view(-1))
            # losses.append(inter_loss)

            self.memory.store(nonceTask[i].item(), nonce_embeds)

        # now do task specific stuff
        b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape

        task_ids = task_inputs["input_ids"].reshape((b_task * k_task, l_task))  # reshape so we have n x seq_len
        task_attn = task_inputs["attention_mask"].reshape((b_task * k_task, l_task))

        if 'task_labels' in batch:
            task_labels = task_labels.reshape((b_task * k_task, l_task))

        new_w = self.get_new_weights(task="Task")
        input_embeds = F.embedding(task_ids, new_w)

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
        return out_vals

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
            probe_out = self.secondLM.roberta(inputs_embeds=probe_input_embeds,
                                              attention_mask=probe_task['attention_mask'].squeeze(0),
                                              output_hidden_states=True)
            probe_embed = probe_out.hidden_states[-1].sum(1)
            probe_embeds.append(probe_embed)
        return nonce_embed, probe_embeds

    def swap_with_mask(self, inputs, k_examples, nonces):
        inp = inputs.clone()
        exp = torch.Tensor(nonces).unsqueeze(1).expand_as(inputs[:, 0, :]).to(
            self.device)  # expand for a set of sentences across batches
        exp = exp.unsqueeze(1).repeat(1, k_examples, 1)  # repeat for k sentences
        inp[inp == exp] = self.mask_token_id

        return inp

    def get_zero_grads_idx(self):
        index_grads_to_zero = torch.empty(self.firstLM.config.vocab_size, dtype=torch.bool).fill_(True)
        for nonce in self.nonces:
            index_grads_to_zero = index_grads_to_zero & (torch.arange(self.firstLM.config.vocab_size) != nonce)
        return index_grads_to_zero

    def initialize_with_def(self, nonce, definition):
        with torch.no_grad():
            toks = self.tokenizer(definition, return_tensors="pt").to(self.device)
            outs = self.firstLM(**toks, output_hidden_states=True)

    def get_new_weights(self, task):

        if task == 'MLM':
            ref_model = self.firstLM

        elif task == "Task":
            ref_model = self.secondLM
        else:
            raise NotImplementedError

        w = ref_model.get_input_embeddings().weight.clone()
        n, hidden = w.shape
        if not ref_model.get_input_embeddings().weight.requires_grad:
            w.requires_grad = True

        msk = torch.zeros_like(w).to(self.device)
        msk2 = torch.zeros_like(w).to(self.device)
        for key in self.memory.memory:
            msk = msk.scatter(0, torch.tensor([key]).to(self.device).expand(1, hidden), self.memory.retrieve(key))
            msk2[key, :] = 1.

        return w * (~msk2.bool()) + msk
        # a = []
        # for i in range(w.shape[0]):
        #     if i in self.memory.memory:
        #         a.append(self.memory.retrieve(i))
        #     else:
        #         a.append(w[i, :].unsqueeze(0))
        # return torch.cat(a, dim=0)

    def update_weights(self):
        first_wt = nn.Embedding(self.firstLM.config.vocab_size,
                                self.firstLM.config.hidden_size).from_pretrained(self.get_new_weights(task='MLM'))
        second_wt = nn.Embedding(self.secondLM.config.vocab_size,
                                 self.secondLM.config.hidden_size).from_pretrained(self.get_new_weights(task='Task'))
        with torch.no_grad():
            self.firstLM.set_input_embeddings(first_wt)

            self.secondLM.set_input_embeddings(second_wt)
        self.freeze()

    def re_encode(self, input_ids, w):

        return torch.nn.functional.embedding(input_ids.squeeze(0), w)

class MorphMemoryModelSQuAD(MorphMemoryModel):

    def __init__(self, firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type):
        super().__init__(firstLM, secondLM, nonces, device, layers, mask_token_id,
                         memory_config, emb_type)
        if self.emb_type == "MLP":
            self.emb_gen = MLP(self.firstLM.config.hidden_size, 384, self.secondLM.config.hidden_size)

        elif self.emb_type == "Transformer":
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.firstLM.config.hidden_size, nhead=2).to(self.device)
            self.emb_gen = nn.TransformerEncoder(encoder_layer, num_layers=1).to(self.device)

    def forward_inner(self, batch):
        mlm_inputs = batch["mlm_inputs"].to(self.device)
        task_inputs = {'input_ids': batch["task_inputs"]['input_ids'].to(self.device),
                       'attention_mask': batch["task_inputs"]['attention_mask'].to(self.device)}
        nonceMLM = batch["nonceMLM"]
        nonceTask = batch["nonceTask"]

        b, k, l = batch["mlm_inputs"]["input_ids"].shape  # batch x k examples x max_length toks

        new_inputs = self.swap_with_mask(mlm_inputs["input_ids"], k, nonceMLM)  # replace nonce with mask

        new_labels = new_inputs.clone()
        new_labels[new_inputs != self.mask_token_id] = -100

        mlm_ids = new_inputs.reshape((b * k, l))  # reshape so we have n x seq_len
        mlm_attn = mlm_inputs["attention_mask"].reshape((b * k, l))

        new_w = self.get_new_weights('Task')

        embs = F.embedding(mlm_ids, new_w)

        outputs = self.secondLM.roberta(
            inputs_embeds=embs,
            attention_mask=mlm_attn,
            output_hidden_states=True
        )

        preds = self.calc_first_lmhead(new_w, outputs[0])
        loss_fct = nn.CrossEntropyLoss()
        lm_loss = loss_fct(preds.view(-1, self.firstLM.config.vocab_size), new_labels.view(-1))
        out_vals = MaskedLMOutput(
            loss=lm_loss,
            logits=preds,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

        return out_vals

    def calc_first_lmhead(self, new_w, last_hidden):
        x = self.firstLM.lm_head.dense(last_hidden)
        x = gelu(x)
        x = self.firstLM.lm_head.layer_norm(x)
        x = F.linear(x, new_w, bias=self.firstLM.lm_head.bias)
        return x

    def forward(self, batch):

        mlm_inputs = batch["mlm_inputs"].to(self.device)
        task_inputs = {'input_ids': batch["task_inputs"]['input_ids'].to(self.device),
                       'attention_mask': batch["task_inputs"]['attention_mask'].to(self.device)}
        nonceMLM = batch["nonceMLM"]
        nonceTask = batch["nonceTask"]
        start_positions = batch['task_start'][0].to(self.device)
        end_positions = batch['task_end'][0].to(self.device)

        b, k, l = batch["mlm_inputs"]["input_ids"].shape  # batch x k examples x max_length toks

        new_inputs = self.swap_with_mask(mlm_inputs["input_ids"], k, nonceMLM)  # replace nonce with mask

        mlm_ids = new_inputs.reshape((b * k, l))  # reshape so we have n x seq_len
        mlm_attn = mlm_inputs["attention_mask"].reshape((b * k, l))

        first_out = self.firstLM(input_ids=mlm_ids, attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        chunked = torch.chunk(combined, b)  # get different embeds per nonce, shape = k x max_len x hidden

        # embedding generator + store in memory
        losses = []
        for i, chunk in enumerate(chunked):
            msk = (new_inputs[i] == self.mask_token_id)  # mask all but the outputs for the mask
            #             if self.emb_type == "Transformer":
            #                 generated_embeds = self.emb_gen(chunk.permute(1, 0, 2), src_key_padding_mask=~msk).permute(1, 0,
            #                                                                                                            2)  # permute for shape, mask, permute back

            embed_ids = msk.nonzero(as_tuple=True)

            if self.emb_type == "Transformer":
                #                 nonce_embeds = generated_embeds[embed_ids[0], embed_ids[1], :]
                nonce_embeds = self.emb_gen(chunk[embed_ids[0], embed_ids[1], :])

            elif self.emb_type == "MLP":
                nonce_embeds = self.emb_gen(chunk[embed_ids[0], embed_ids[1], :])

            # preds = self.emb_decoder(nonce_embeds)
            # labs = nonceMLM[i].to(self.device).repeat(preds.shape[0])
            # l_fct = nn.CrossEntropyLoss()
            # inter_loss = l_fct(preds.view(-1, self.firstLM.config.vocab_size), labs.view(-1))
            # losses.append(inter_loss)
            #             print((~msk).nonzero()[:,0].tolist())
            #             if nonce_embeds.shape[0] > 2:
            #                 print(batch['mlm_inputs']['input_ids'])
            #                 print(embed_ids)
            #                 print('shape', batch['mlm_inputs']['input_ids'].shape)
            #                 print('nonce shape', nonce_embeds.shape)
            #                 raise Exception()
            self.memory.store(nonceMLM[i].item(), nonce_embeds)

        b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape

        task_ids = task_inputs["input_ids"].reshape((b_task * k_task, l_task))  # reshape so we have n x seq_len
        task_attn = task_inputs["attention_mask"].reshape((b_task * k_task, l_task))

        new_w = self.get_new_weights(task="Task")

        input_embeds = F.embedding(task_ids, new_w)

        outputs = self.secondLM.roberta(
            inputs_embeds=input_embeds,
            attention_mask=task_attn,
            output_hidden_states=True
        )

        sequence_output = outputs[0]

        logits = self.secondLM.qa_outputs(sequence_output)
        #         print(logits)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None

        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        out_vals = QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        return out_vals

class MorphMemoryModelSNLI(MorphMemoryModel):

    def __init__(self, firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_gen):
        super(MorphMemoryModelSNLI, self).__init__(firstLM, secondLM, nonces, device, layers, mask_token_id,
                                                   memory_config, emb_gen)
    def store_mem(self, batch):
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

        new_inputs = self.swap_with_mask(mlm_inputs["input_ids"], k, nonceMLM)  # replace nonce with mask

        mlm_ids = new_inputs.reshape((b * k, l))  # reshape so we have n x seq_len
        mlm_attn = mlm_inputs["attention_mask"].reshape((b * k, l))

        first_out = self.firstLM(input_ids=mlm_ids, attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        chunked = torch.chunk(combined, b)  # get different embeds per nonce, shape = k x max_len x hidden

        # embedding generator + store in memory
        losses = []
        for i, chunk in enumerate(chunked):
            msk = (new_inputs[i] == self.mask_token_id)  # mask all but the outputs for the mask
            if self.emb_type == "Transformer":
                generated_embeds = self.emb_gen(chunk.permute(1, 0, 2), src_key_padding_mask=~msk).permute(1, 0,
                                                                                                           2)  # permute for shape, mask, permute back

            embed_ids = msk.nonzero(as_tuple=True)

            if self.emb_type == "Transformer":
                nonce_embeds = generated_embeds[embed_ids[0], embed_ids[1], :]

            elif self.emb_type == "MLP":
                nonce_embeds = self.emb_gen(chunk[embed_ids[0], embed_ids[1], :])

            # preds = self.emb_decoder(nonce_embeds)
            # labs = nonceMLM[i].to(self.device).repeat(preds.shape[0])
            # l_fct = nn.CrossEntropyLoss()
            # inter_loss = l_fct(preds.view(-1, self.firstLM.config.vocab_size), labs.view(-1))
            # losses.append(inter_loss)

            self.memory.store(nonceMLM[i].item(), nonce_embeds)

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

        new_inputs = self.swap_with_mask(mlm_inputs["input_ids"], k, nonceMLM)  # replace nonce with mask

        mlm_ids = new_inputs.reshape((b * k, l))  # reshape so we have n x seq_len
        mlm_attn = mlm_inputs["attention_mask"].reshape((b * k, l))

        first_out = self.firstLM(input_ids=mlm_ids, attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        chunked = torch.chunk(combined, b)  # get different embeds per nonce, shape = k x max_len x hidden

        # embedding generator + store in memory
        losses = []
        for i, chunk in enumerate(chunked):
            msk = (new_inputs[i] == self.mask_token_id)  # mask all but the outputs for the mask
            if self.emb_type == "Transformer":
                generated_embeds = self.emb_gen(chunk.permute(1, 0, 2), src_key_padding_mask=~msk).permute(1, 0,
                                                                                                           2)  # permute for shape, mask, permute back

            embed_ids = msk.nonzero(as_tuple=True)

            if self.emb_type == "Transformer":
                nonce_embeds = generated_embeds[embed_ids[0], embed_ids[1], :]

            elif self.emb_type == "MLP":
                nonce_embeds = self.emb_gen(chunk[embed_ids[0], embed_ids[1], :])

            # preds = self.emb_decoder(nonce_embeds)
            # labs = nonceMLM[i].to(self.device).repeat(preds.shape[0])
            # l_fct = nn.CrossEntropyLoss()
            # inter_loss = l_fct(preds.view(-1, self.firstLM.config.vocab_size), labs.view(-1))
            # losses.append(inter_loss)

            self.memory.store(nonceMLM[i].item(), nonce_embeds)

        b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape

        task_ids = task_inputs["input_ids"].reshape((b_task * k_task, l_task))  # reshape so we have n x seq_len
        task_attn = task_inputs["attention_mask"].reshape((b_task * k_task, l_task))

        new_w = self.get_new_weights(task="Task")
        input_embeds = F.embedding(task_ids, new_w)
        out_vals = self.secondLM(
            inputs_embeds=input_embeds,
            attention_mask=task_attn,
            labels=task_labels.squeeze(1),
            output_hidden_states=True
        )

        return out_vals

    def forward_inner(self, batch):
        mlm_inputs = batch["mlm_inputs"].to(self.device)
        nonceMLM = batch["nonceMLM"]

        b, k, l = batch["mlm_inputs"]["input_ids"].shape  # batch x k examples x max_length toks

        new_inputs = self.swap_with_mask(mlm_inputs["input_ids"], k, nonceMLM)  # replace nonce with mask

        new_labels = new_inputs.clone()
        new_labels[new_inputs != self.mask_token_id] = -100

        mlm_ids = new_inputs.reshape((b * k, l))  # reshape so we have n x seq_len
        mlm_attn = mlm_inputs["attention_mask"].reshape((b * k, l))

        new_w = self.get_new_weights('Task')

        embs = F.embedding(mlm_ids, new_w)

        outputs = self.secondLM.roberta(
            inputs_embeds=embs,
            attention_mask=mlm_attn,
            output_hidden_states=True
        )

        preds = self.calc_first_lmhead(new_w, outputs[0])
        loss_fct = nn.CrossEntropyLoss()
        lm_loss = loss_fct(preds.view(-1, self.firstLM.config.vocab_size), new_labels.view(-1))
        out_vals = MaskedLMOutput(
            loss=lm_loss,
            logits=preds,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )

        return out_vals

    def calc_first_lmhead(self, new_w, last_hidden):
        x = self.firstLM.lm_head.dense(last_hidden)
        x = gelu(x)
        x = self.firstLM.lm_head.layer_norm(x)
        x = F.linear(x, new_w, bias=self.firstLM.lm_head.bias)
        return x

class MorphMemoryModelGPT(MorphMemoryModel):

    def __init__(self, firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type):
        super().__init__(firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type)


    def forward(self, batch):

        mlm_inputs = batch["mlm_inputs"].to(self.device)
        task_inputs = {'input_ids': batch["task_inputs"]['input_ids'].to(self.device),
                       'attention_mask': batch["task_inputs"]['attention_mask'].to(self.device)}
        nonceMLM = batch["nonceMLM"]
        nonceTask = batch['nonceTask']
        if 'task_labels' in batch:
            task_labels = batch["task_labels"].to(self.device)

        else:
            task_labels = None

        b, k, l = batch["mlm_inputs"]["input_ids"].shape  # batch x k examples x max_length toks

        new_inputs = self.swap_with_mask(mlm_inputs["input_ids"], k, nonceMLM)  # replace nonce with mask

        mlm_ids = new_inputs.reshape((b * k, l))  # reshape so we have n x seq_len
        mlm_attn = mlm_inputs["attention_mask"].reshape((b * k, l))

        first_out = self.firstLM(input_ids=mlm_ids, attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        chunked = torch.chunk(combined, b)  # get different embeds per nonce, shape = k x max_len x hidden

        # embedding generator + store in memory
        losses = []
        for i, chunk in enumerate(chunked):
            msk = (new_inputs[i] == self.mask_token_id)  # mask all but the outputs for the mask
            #             if self.emb_type == "Transformer":
            #                 generated_embeds = self.emb_gen(chunk.permute(1, 0, 2), src_key_padding_mask=~msk).permute(1, 0,
            #                                                                                                            2)  # permute for shape, mask, permute back

            embed_ids = msk.nonzero(as_tuple=True)

            #             if self.emb_type == "Transformer":
            #                 nonce_embeds = generated_embeds[embed_ids[0], embed_ids[1], :]

            #             elif self.emb_type == "MLP":
            nonce_embeds = self.emb_gen(chunk[embed_ids[0], embed_ids[1], :])

            self.memory.store(nonceTask[i].item(), nonce_embeds)

        # now do task specific stuff
        b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape

        task_ids = task_inputs["input_ids"].reshape((b_task * k_task, l_task))  # reshape so we have n x seq_len
        task_attn = task_inputs["attention_mask"].reshape((b_task * k_task, l_task))

        if 'task_labels' in batch:
            task_labels = task_labels.reshape((b_task * k_task, l_task))

        new_w = self.get_new_weights(task="Task")
        input_embeds = F.embedding(task_ids, new_w)

        outputs = self.secondLM.transformer(
            inputs_embeds=input_embeds,
            attention_mask=task_attn,
            output_hidden_states=True
        )

        lm_logits = F.linear(outputs[0], new_w)
        loss_fct = nn.CrossEntropyLoss()

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = task_labels[..., 1:].contiguous()

        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # #         l = nn.CrossEntropyLoss(reduction="none")

        out_vals = CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
        return out_vals

    def swap_with_mask(self, inputs, k_examples, nonces):
        inp = inputs.clone()
        exp = torch.Tensor(nonces).unsqueeze(1).expand_as(inputs[:, 0, :]).to(
            self.device)  # expand for a set of sentences across batches
        exp = exp.unsqueeze(1).repeat(1, k_examples, 1)  # repeat for k sentences
        inp[inp == exp] = self.mask_token_id

        return inp

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, do_sample=False):
        idx = idx.to(self.device)
        for i in range(max_new_tokens):
            new_w = self.get_new_weights("Task")
            input_embeds = F.embedding(idx['input_ids'], new_w)
            outputs = self.secondLM.transformer(
                inputs_embeds=input_embeds,
                attention_mask=idx['attention_mask'].unsqueeze(0),
                output_hidden_states=True
            )
            lm_logits = F.linear(outputs[0], new_w)
            lm_logits = lm_logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(lm_logits, min(top_k, lm_logits.size(-1)))
                lm_logits[lm_logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(lm_logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, keepdim=True)

            idx['input_ids'] = torch.cat((idx['input_ids'], idx_next), dim=1)
            last_element = idx['attention_mask'][:, -1].unsqueeze(1)
            idx['attention_mask'] = torch.cat([idx['attention_mask'], last_element], dim=1)

        return idx

    def re_encode(self, input_ids, w):

        return torch.nn.functional.embedding(input_ids.squeeze(0), w)


class MorphMemoryModelGPTOnline(MorphMemoryModelGPT):

    def __init__(self, firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type):
        super().__init__(firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type)
        initial_first_ind = int(self.firstLM.config.vocab_size - len(self.nonces))
        initial_second_ind = int(self.secondLM.config.vocab_size - len(self.nonces))

        self.first_list = list(range(initial_first_ind, self.firstLM.config.vocab_size))
        self.second_list = list(range(initial_second_ind, self.secondLM.config.vocab_size))

        self.model_name = "online_memory_model_{}_{}_{}_memory".format(self.firstLM.config.model_type,
                                                                       self.secondLM.config.model_type,
                                                                       memory_config.agg_method)

    def process_memories(self, mem):

        b, l = mem["input_ids"].shape
        mlm_ids = self.swap_with_mask(mem['input_ids'].to(self.device))
        #         mlm_ids = new_inputs.reshape((b * k, l))
        mlm_attn = mem["attention_mask"].to(self.device)
        first_out = self.firstLM(input_ids=mlm_ids,
                                 attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        for nonce1, nonce2 in zip(self.first_list, self.second_list):
            if nonce1 in mem["input_ids"]:
                msk = (mem["input_ids"] == nonce1)

                embed_ids = msk.nonzero(as_tuple=True)
                if len(combined.shape) == 2:
                    nonce_embeds = self.emb_gen(combined[embed_ids[1], :])
                elif len(combined.shape) == 3:
                    nonce_embeds = self.emb_gen(combined[embed_ids[0], embed_ids[1], :])
                else:
                    raise NotImplementedError("Wrong shape for Combined")
                self.memory.store(nonce2, nonce_embeds)

    def swap_with_mask(self, inputs):
        inp = inputs.clone()
        for nonce in self.first_list:
            inp[inp == nonce] = self.mask_token_id
        return inp

    def forward(self, batch):

        mlm_inputs = batch["mlm_inputs"].to(self.device)
        task_inputs = {'input_ids': batch["task_inputs"]['input_ids'].to(self.device),
                       'attention_mask': batch["task_inputs"]['attention_mask'].to(self.device)}
        if 'task_labels' in batch:
            task_labels = batch["task_labels"].to(self.device)

        else:
            task_labels = None

        b, k, l = batch["mlm_inputs"]["input_ids"].shape  # batch x k examples x max_length toks

        new_inputs = self.swap_with_mask(mlm_inputs["input_ids"])  # replace nonce with mask

        mlm_ids = new_inputs.reshape((b * k, l))  # reshape so we have n x seq_len
        mlm_attn = mlm_inputs["attention_mask"].reshape((b * k, l))

        first_out = self.firstLM(input_ids=mlm_ids, attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        chunked = torch.chunk(combined, b)  # get different embeds per nonce, shape = k x max_len x hidden

        # embedding generator + store in memory
        losses = []
        for nonce1, nonce2 in zip(self.first_list, self.second_list):
            if nonce1 in mlm_inputs["input_ids"]:
                msk = (mlm_inputs["input_ids"].reshape((b * k, l)) == nonce1)

                embed_ids = msk.nonzero(as_tuple=True)
                nonce_embeds = self.emb_gen(combined[embed_ids[0], embed_ids[1], :])

                self.memory.store(nonce2, nonce_embeds)

        # now do task specific stuff
        b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape

        task_ids = task_inputs["input_ids"].reshape((b_task * k_task, l_task))  # reshape so we have n x seq_len
        task_attn = task_inputs["attention_mask"].reshape((b_task * k_task, l_task))

        if 'task_labels' in batch:
            task_labels = task_labels.reshape((b_task * k_task, l_task))

        new_w = self.get_new_weights(task="Task")
        input_embeds = F.embedding(task_ids, new_w)

        outputs = self.secondLM.transformer(
            inputs_embeds=input_embeds,
            attention_mask=task_attn,
            output_hidden_states=True
        )

        lm_logits = F.linear(outputs[0], new_w)
        loss_fct = nn.CrossEntropyLoss()

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = task_labels[..., 1:].contiguous()

        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # #         l = nn.CrossEntropyLoss(reduction="none")

        out_vals = CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
        return out_vals

class MorphMemoryModelGPTSubtoken(MorphMemoryModel):

    def __init__(self, firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type):
        super().__init__(firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type)
        initial_first_ind = int(self.firstLM.config.vocab_size - len(self.nonces))
        initial_second_ind = int(self.secondLM.config.vocab_size - len(self.nonces))

        self.first_list = list(range(initial_first_ind, self.firstLM.config.vocab_size))
        self.second_list = list(range(initial_second_ind, self.secondLM.config.vocab_size))

        self.pe = PositionalEncoding(self.firstLM.config.hidden_size, self.firstLM.config.max_position_embeddings)

    def make_mask(self, ids, spans):

        mask = torch.zeros_like(ids).to(self.device)
        for i, span in enumerate(spans):
            mask[i, span] = 1.

        return mask

    def forward(self, batch):

        mlm_inputs = batch["mlm_inputs"].to(self.device)
        task_inputs = {'input_ids': batch["task_inputs"]['input_ids'].to(self.device),
                       'attention_mask': batch["task_inputs"]['attention_mask'].to(self.device)}
        firstSpan = batch["firstSpan"]
        secondSpan = batch['secondSpan']

        if 'task_labels' in batch:
            task_labels = batch["task_labels"].to(self.device)

        else:
            task_labels = None

        b, k, l = batch["mlm_inputs"]["input_ids"].shape  # batch x k examples x max_length toks

        #         new_inputs = self.swap_with_mask(mlm_inputs["input_ids"], k, nonceMLM)  # replace nonce with mask

        mlm_ids = mlm_inputs['input_ids'].reshape((b * k, l))  # reshape so we have n x seq_len
        mlm_attn = mlm_inputs["attention_mask"].reshape((b * k, l))

        first_out = self.firstLM(input_ids=mlm_ids, attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        combined = self.pe(combined)
        # embedding generator + store in memory
        losses = []
        spans = [firstSpan, secondSpan]
        for i, nonce in enumerate(self.second_list):
            msk = self.make_mask(mlm_ids, spans[i])
            if self.emb_type == "Transformer":
                generated_embeds = self.emb_gen(combined.permute(1, 0, 2), src_key_padding_mask=~msk.bool()).permute(1,
                                                                                                                     0,
                                                                                                                     2)  # permute for shape, mask, permute back
            else:
                raise NotImplementedError

            # mean pooling per seq
            nonce_embeds = (generated_embeds * msk.unsqueeze(2)).sum(dim=1) / msk.unsqueeze(2).sum(dim=1)

            self.memory.store(nonce, nonce_embeds)

        # now do task specific stuff
        b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape

        task_ids = task_inputs["input_ids"].reshape((b_task * k_task, l_task))  # reshape so we have n x seq_len
        task_attn = task_inputs["attention_mask"].reshape((b_task * k_task, l_task))

        if 'task_labels' in batch:
            task_labels = task_labels.reshape((b_task * k_task, l_task))

        new_w = self.get_new_weights(task="Task")
        input_embeds = F.embedding(task_ids, new_w)

        outputs = self.secondLM.transformer(
            inputs_embeds=input_embeds,
            attention_mask=task_attn,
            output_hidden_states=True
        )

        lm_logits = F.linear(outputs[0], new_w)
        loss_fct = nn.CrossEntropyLoss()

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = task_labels[..., 1:].contiguous()

        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # #         l = nn.CrossEntropyLoss(reduction="none")

        out_vals = CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
        return out_vals

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, do_sample=False):
        idx = idx.to(self.device)
        for i in range(max_new_tokens):
            new_w = self.get_new_weights("Task")
            input_embeds = F.embedding(idx['input_ids'], new_w)
            outputs = self.secondLM.transformer(
                inputs_embeds=input_embeds,
                attention_mask=idx['attention_mask'].unsqueeze(0),
                output_hidden_states=True
            )
            lm_logits = F.linear(outputs[0], new_w)
            lm_logits = lm_logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(lm_logits, min(top_k, lm_logits.size(-1)))
                lm_logits[lm_logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(lm_logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, keepdim=True)

            idx['input_ids'] = torch.cat((idx['input_ids'], idx_next), dim=1)
            last_element = idx['attention_mask'][:, -1].unsqueeze(1)
            idx['attention_mask'] = torch.cat([idx['attention_mask'], last_element], dim=1)

        return idx

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class MorphMemoryModelMLMOnline(MorphMemoryModel):

    def __init__(self, firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type):
        super().__init__(firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type)
        initial_first_ind = int(self.firstLM.config.vocab_size - len(self.nonces))
        initial_second_ind = int(self.secondLM.config.vocab_size - len(self.nonces))

        self.first_list = list(range(initial_first_ind, self.firstLM.config.vocab_size))
        self.second_list = list(range(initial_second_ind, self.secondLM.config.vocab_size))

        self.model_name = "MLMonline_memory_model_{}_{}_{}_memory".format(self.firstLM.config.model_type,
                                                                       self.secondLM.config.model_type,
                                                                       memory_config.agg_method)

    def process_memories(self, mem):

        b, l = mem["input_ids"].shape
        mlm_ids = self.swap_with_mask(mem['input_ids'].to(self.device))
        #         mlm_ids = new_inputs.reshape((b * k, l))
        mlm_attn = mem["attention_mask"].to(self.device)
        first_out = self.firstLM(input_ids=mlm_ids,
                                 attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        for nonce1, nonce2 in zip(self.first_list, self.second_list):
            if nonce1 in mem["input_ids"]:
                msk = (mem["input_ids"] == nonce1)

                embed_ids = msk.nonzero(as_tuple=True)
                if len(combined.shape) == 2:
                    nonce_embeds = self.emb_gen(combined[embed_ids[1], :])
                elif len(combined.shape) == 3:
                    nonce_embeds = self.emb_gen(combined[embed_ids[0], embed_ids[1], :])
                else:
                    raise NotImplementedError("Wrong shape for Combined")
                nonce_embeds = torch.mean(nonce_embeds, dim=0, keepdim=True)
                self.memory.store(nonce2, nonce_embeds)

    def swap_with_mask(self, inputs):
        inp = inputs.clone()
        for nonce in self.first_list:
            inp[inp == nonce] = self.mask_token_id
        return inp

    def forward(self, batch):

        mlm_inputs = batch["mlm_inputs"].to(self.device)
        task_inputs = {'input_ids': batch["task_inputs"]['input_ids'].to(self.device),
                       'attention_mask': batch["task_inputs"]['attention_mask'].to(self.device)}
        if 'task_labels' in batch:
            task_labels = batch["task_labels"].to(self.device)

        else:
            task_labels = None

        b, k, l = batch["mlm_inputs"]["input_ids"].shape  # batch x k examples x max_length toks

        new_inputs = self.swap_with_mask(mlm_inputs["input_ids"])  # replace nonce with mask

        mlm_ids = new_inputs.reshape((b * k, l))  # reshape so we have n x seq_len
        mlm_attn = mlm_inputs["attention_mask"].reshape((b * k, l))

        first_out = self.firstLM(input_ids=mlm_ids, attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        chunked = torch.chunk(combined, b)  # get different embeds per nonce, shape = k x max_len x hidden

        # embedding generator + store in memory
        losses = []
        for nonce1, nonce2 in zip(self.first_list, self.second_list):
            if nonce1 in mlm_inputs["input_ids"]:
                msk = (mlm_inputs["input_ids"].reshape((b * k, l)) == nonce1)

                embed_ids = msk.nonzero(as_tuple=True)
                nonce_embeds = self.emb_gen(combined[embed_ids[0], embed_ids[1], :])
                nonce_embeds = torch.mean(nonce_embeds, dim=0, keepdim=True)
                self.memory.store(nonce2, nonce_embeds)

        # now do task specific stuff
        b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape

        task_ids = task_inputs["input_ids"].reshape((b_task * k_task, l_task))  # reshape so we have n x seq_len
        task_attn = task_inputs["attention_mask"].reshape((b_task * k_task, l_task))

        if 'task_labels' in batch:
            task_labels = task_labels.reshape((b_task * k_task, l_task))

        new_w = self.get_new_weights(task="Task")
        input_embeds = F.embedding(task_ids, new_w)

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
        return out_vals


class MorphMemoryModelMLMOnlineBinary(MorphMemoryModel):

    def __init__(self, firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type):
        super().__init__(firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.firstLM.config.hidden_size,
                                                   nhead=self.firstLM.config.num_attention_heads,
                                                   activation='gelu',
                                                   batch_first=True).to(self.device)
        self.emb_gen = nn.TransformerEncoder(encoder_layer, num_layers=1).to(self.device)
        initial_first_ind = int(self.firstLM.config.vocab_size - len(self.nonces))
        initial_second_ind = int(self.secondLM.config.vocab_size - len(self.nonces))

        self.first_list = list(range(initial_first_ind, self.firstLM.config.vocab_size))
        self.second_list = list(range(initial_second_ind, self.secondLM.config.vocab_size))

        self.model_name = "MLMonline_memory_model_{}_{}_{}_memory".format(self.firstLM.config.model_type,
                                                                          self.secondLM.config.model_type,
                                                                          memory_config.agg_method)
        self.pos_binary = nn.Parameter(torch.randn(1, self.firstLM.config.hidden_size, device=self.device))
        self.neg_binary = nn.Parameter(torch.randn(1, self.firstLM.config.hidden_size, device=self.device))

        self.cls_token = nn.Parameter(torch.randn(1, self.firstLM.config.hidden_size, device=self.device))

    def process_memories(self, mem):

        b, l = mem["input_ids"].shape
        mlm_ids = self.swap_with_mask(mem['input_ids'].to(self.device))
        #         mlm_ids = new_inputs.reshape((b * k, l))
        mlm_attn = mem["attention_mask"].to(self.device)
        first_out = self.firstLM(input_ids=mlm_ids,
                                 attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        for nonce1, nonce2 in zip(self.first_list, self.second_list):
            if nonce1 in mem["input_ids"]:
                #                 msk = (mem["input_ids"] == nonce1)

                #                 embed_ids = msk.nonzero(as_tuple=True)
                #                 if len(combined.shape) == 2:
                #                     nonce_embeds = self.emb_gen(combined[embed_ids[1], :])
                #                 elif len(combined.shape) == 3:
                #                     nonce_embeds = self.emb_gen(combined[embed_ids[0], embed_ids[1], :])
                #                 else:
                #                     raise NotImplementedError("Wrong shape for Combined")
                #                 print(nonce_embeds.shape, "before")
                #                 nonce_embeds = torch.mean(nonce_embeds, dim=0, keepdim=True)
                #                 print(nonce_embeds.shape, "after")
                #                 self.memory.store(nonce2, nonce_embeds)

                src = mem["input_ids"]
                src = src.unsqueeze(-1)
                src = src.expand(-1, -1, self.firstLM.config.hidden_size)
                pos = self.pos_binary.unsqueeze(0).expand(src.shape[0], -1, -1)
                neg = self.neg_binary.unsqueeze(0).expand(src.shape[0], -1, -1)
                mask = src == nonce1
                added = torch.where(mask.to(self.device), pos, neg)
                cls = self.cls_token.unsqueeze(0).expand(src.shape[0], -1, -1)
                #                 return added, msk
                embed_inputs = combined + added

                embed_inputs = torch.cat([cls, embed_inputs], dim=1)
                nonce_embeds = self.emb_gen(embed_inputs)
                self.memory.store(nonce2, nonce_embeds[:, 0])

    def swap_with_mask(self, inputs):
        inp = inputs.clone()
        for nonce in self.first_list:
            inp[inp == nonce] = self.mask_token_id
        return inp

    def forward(self, batch):

        mlm_inputs = batch["mlm_inputs"].to(self.device)
        task_inputs = {'input_ids': batch["task_inputs"]['input_ids'].to(self.device),
                       'attention_mask': batch["task_inputs"]['attention_mask'].to(self.device)}
        if 'task_labels' in batch:
            task_labels = batch["task_labels"].to(self.device)

        else:
            task_labels = None

        b, k, l = batch["mlm_inputs"]["input_ids"].shape  # batch x k examples x max_length toks

        new_inputs = self.swap_with_mask(mlm_inputs["input_ids"])  # replace nonce with mask

        mlm_ids = new_inputs.reshape((b * k, l))  # reshape so we have n x seq_len
        mlm_attn = mlm_inputs["attention_mask"].reshape((b * k, l))

        first_out = self.firstLM(input_ids=mlm_ids, attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        chunked = torch.chunk(combined, b)  # get different embeds per nonce, shape = k x max_len x hidden

        # embedding generator + store in memory
        losses = []
        for nonce1, nonce2 in zip(self.first_list, self.second_list):
            if nonce1 in mlm_inputs["input_ids"]:
                msk = (mlm_inputs["input_ids"].reshape((b * k, l)) == nonce1)
                src = mlm_inputs["input_ids"].reshape((b * k, l))[msk.nonzero()[:, 0].unique()]
                src = src.unsqueeze(-1)
                src = src.expand(-1, -1, self.firstLM.config.hidden_size)
                pos = self.pos_binary.unsqueeze(0).expand(src.shape[0], -1, -1)
                neg = self.neg_binary.unsqueeze(0).expand(src.shape[0], -1, -1)
                mask = src == nonce1
                added = torch.where(mask, pos, neg)
                cls = self.cls_token.unsqueeze(0).expand(src.shape[0], -1, -1)
                #                 return added, msk
                embed_inputs = combined[msk.nonzero()[:, 0].unique()] + added

                embed_inputs = torch.cat([cls, embed_inputs], dim=1)
                nonce_embeds = self.emb_gen(embed_inputs)
                self.memory.store(nonce2, nonce_embeds[:, 0])
        #                 return added
        #                 src_mask = torch.einsum("bj, ib->bij", msk.float(), msk.float().T).bool()
        # #                 src_mask = (msk.float().T @ msk.float()).bool()
        #                 print(src_mask)
        #                 print(src_mask.shape)
        #                 print(src_mask.nonzero())
        #                 print(msk.nonzero())
        #                 expanded = src_mask.expand(6, src_mask.shape[0], src_mask.shape[1], src_mask.shape[2])
        #                 print(expanded.shape)
        #                 print(6* src_mask.shape[0], src_mask.shape[1], src_mask.shape[2])
        #                 per_head_mask = expanded.contiguous().view(6* src_mask.shape[0], src_mask.shape[1], src_mask.shape[2])
        # #                 src_mask = torch.ones(msk.shape[1], msk.shape[1], device=self.device)
        # #                 zeros = torch.zeros(msk.shape[1], msk.shape[1], device=self.device)
        # #                 src_mask = torch.where(msk, zeros, torch.ones(1,1, dtype=bool, device=self.device))
        # #                 src_mask[torch.ara]
        # #                 print(src_mask, src_mask.shape, (~src_mask).nonzero())

        #                 embed_ids = msk.nonzero(as_tuple=True)
        #                 print(combined.shape)
        # #                 nonce_embeds = self.emb_gen(combined[embed_ids[0], embed_ids[1], :])
        #                 nonce_embeds = self.emb_gen(combined, mask=~per_head_mask)
        #                 return src_mask, combined, per_head_mask, nonce_embeds
        #                 print(nonce_embeds)
        #                 print(nonce_embeds.shape, "before")
        #                 nonce_embeds = torch.mean(nonce_embeds, dim=0, keepdim=True)
        #                 print(nonce_embeds.shape, "after")
        #                 self.memory.store(nonce2, nonce_embeds)

        # now do task specific stuff
        b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape

        task_ids = task_inputs["input_ids"].reshape((b_task * k_task, l_task))  # reshape so we have n x seq_len
        task_attn = task_inputs["attention_mask"].reshape((b_task * k_task, l_task))

        if 'task_labels' in batch:
            task_labels = task_labels.reshape((b_task * k_task, l_task))

        new_w = self.get_new_weights(task="Task")
        input_embeds = F.embedding(task_ids, new_w)

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
        return out_vals


class MorphMemoryModelGPTOnlineBinary(MorphMemoryModel):

    def __init__(self, firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type):
        super().__init__(firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.firstLM.config.hidden_size,
                                                   nhead=self.firstLM.config.num_attention_heads,
                                                   activation='gelu',
                                                   batch_first=True).to(self.device)
        self.emb_gen = nn.TransformerEncoder(encoder_layer, num_layers=1).to(self.device)
        initial_first_ind = int(self.firstLM.config.vocab_size - len(self.nonces))
        initial_second_ind = int(self.secondLM.config.vocab_size - len(self.nonces))

        self.first_list = list(range(initial_first_ind, self.firstLM.config.vocab_size))
        self.second_list = list(range(initial_second_ind, self.secondLM.config.vocab_size))

        self.model_name = "MLMonline_memory_model_{}_{}_{}_memory".format(self.firstLM.config.model_type,
                                                                          self.secondLM.config.model_type,
                                                                          memory_config.agg_method)
        self.pos_binary = nn.Parameter(torch.randn(1, self.firstLM.config.hidden_size, device=self.device))
        self.neg_binary = nn.Parameter(torch.randn(1, self.firstLM.config.hidden_size, device=self.device))

        self.cls_token = nn.Parameter(torch.randn(1, self.firstLM.config.hidden_size, device=self.device))

    def process_memories(self, mem):

        b, l = mem["input_ids"].shape
        mlm_ids = self.swap_with_mask(mem['input_ids'].to(self.device))
        #         mlm_ids = new_inputs.reshape((b * k, l))
        mlm_attn = mem["attention_mask"].to(self.device)
        first_out = self.firstLM(input_ids=mlm_ids,
                                 attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        for nonce1, nonce2 in zip(self.first_list, self.second_list):
            if nonce1 in mem["input_ids"]:
                #                 msk = (mem["input_ids"] == nonce1)

                #                 embed_ids = msk.nonzero(as_tuple=True)
                #                 if len(combined.shape) == 2:
                #                     nonce_embeds = self.emb_gen(combined[embed_ids[1], :])
                #                 elif len(combined.shape) == 3:
                #                     nonce_embeds = self.emb_gen(combined[embed_ids[0], embed_ids[1], :])
                #                 else:
                #                     raise NotImplementedError("Wrong shape for Combined")
                #                 print(nonce_embeds.shape, "before")
                #                 nonce_embeds = torch.mean(nonce_embeds, dim=0, keepdim=True)
                #                 print(nonce_embeds.shape, "after")
                #                 self.memory.store(nonce2, nonce_embeds)

                src = mem["input_ids"]
                src = src.unsqueeze(-1)
                src = src.expand(-1, -1, self.firstLM.config.hidden_size)
                pos = self.pos_binary.unsqueeze(0).expand(src.shape[0], -1, -1)
                neg = self.neg_binary.unsqueeze(0).expand(src.shape[0], -1, -1)
                mask = src == nonce1
                added = torch.where(mask.to(self.device), pos, neg)
                cls = self.cls_token.unsqueeze(0).expand(src.shape[0], -1, -1)
                #                 return added, msk
                embed_inputs = combined + added

                embed_inputs = torch.cat([cls, embed_inputs], dim=1)
                nonce_embeds = self.emb_gen(embed_inputs)
                self.memory.store(nonce2, nonce_embeds[:, 0])

    def swap_with_mask(self, inputs):
        inp = inputs.clone()
        for nonce in self.first_list:
            inp[inp == nonce] = self.mask_token_id
        return inp

    def forward(self, batch):

        mlm_inputs = batch["mlm_inputs"].to(self.device)
        task_inputs = {'input_ids': batch["task_inputs"]['input_ids'].to(self.device),
                       'attention_mask': batch["task_inputs"]['attention_mask'].to(self.device)}
        if 'task_labels' in batch:
            task_labels = batch["task_labels"].to(self.device)

        else:
            task_labels = None

        b, k, l = batch["mlm_inputs"]["input_ids"].shape  # batch x k examples x max_length toks

        new_inputs = self.swap_with_mask(mlm_inputs["input_ids"])  # replace nonce with mask

        mlm_ids = new_inputs.reshape((b * k, l))  # reshape so we have n x seq_len
        mlm_attn = mlm_inputs["attention_mask"].reshape((b * k, l))

        first_out = self.firstLM(input_ids=mlm_ids, attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        chunked = torch.chunk(combined, b)  # get different embeds per nonce, shape = k x max_len x hidden

        # embedding generator + store in memory
        losses = []
        for nonce1, nonce2 in zip(self.first_list, self.second_list):
            if nonce1 in mlm_inputs["input_ids"]:
                msk = (mlm_inputs["input_ids"].reshape((b * k, l)) == nonce1)
                src = mlm_inputs["input_ids"].reshape((b * k, l))[msk.nonzero()[:, 0].unique()]
                src = src.unsqueeze(-1)
                src = src.expand(-1, -1, self.firstLM.config.hidden_size)
                pos = self.pos_binary.unsqueeze(0).expand(src.shape[0], -1, -1)
                neg = self.neg_binary.unsqueeze(0).expand(src.shape[0], -1, -1)
                mask = src == nonce1
                added = torch.where(mask, pos, neg)
                cls = self.cls_token.unsqueeze(0).expand(src.shape[0], -1, -1)
                #                 return added, msk
                embed_inputs = combined[msk.nonzero()[:, 0].unique()] + added

                embed_inputs = torch.cat([cls, embed_inputs], dim=1)
                nonce_embeds = self.emb_gen(embed_inputs)
                self.memory.store(nonce2, nonce_embeds[:, 0])
        #                 return added
        #                 src_mask = torch.einsum("bj, ib->bij", msk.float(), msk.float().T).bool()
        # #                 src_mask = (msk.float().T @ msk.float()).bool()
        #                 print(src_mask)
        #                 print(src_mask.shape)
        #                 print(src_mask.nonzero())
        #                 print(msk.nonzero())
        #                 expanded = src_mask.expand(6, src_mask.shape[0], src_mask.shape[1], src_mask.shape[2])
        #                 print(expanded.shape)
        #                 print(6* src_mask.shape[0], src_mask.shape[1], src_mask.shape[2])
        #                 per_head_mask = expanded.contiguous().view(6* src_mask.shape[0], src_mask.shape[1], src_mask.shape[2])
        # #                 src_mask = torch.ones(msk.shape[1], msk.shape[1], device=self.device)
        # #                 zeros = torch.zeros(msk.shape[1], msk.shape[1], device=self.device)
        # #                 src_mask = torch.where(msk, zeros, torch.ones(1,1, dtype=bool, device=self.device))
        # #                 src_mask[torch.ara]
        # #                 print(src_mask, src_mask.shape, (~src_mask).nonzero())

        #                 embed_ids = msk.nonzero(as_tuple=True)
        #                 print(combined.shape)
        # #                 nonce_embeds = self.emb_gen(combined[embed_ids[0], embed_ids[1], :])
        #                 nonce_embeds = self.emb_gen(combined, mask=~per_head_mask)
        #                 return src_mask, combined, per_head_mask, nonce_embeds
        #                 print(nonce_embeds)
        #                 print(nonce_embeds.shape, "before")
        #                 nonce_embeds = torch.mean(nonce_embeds, dim=0, keepdim=True)
        #                 print(nonce_embeds.shape, "after")
        #                 self.memory.store(nonce2, nonce_embeds)

        # now do task specific stuff
        b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape

        task_ids = task_inputs["input_ids"].reshape((b_task * k_task, l_task))  # reshape so we have n x seq_len
        task_attn = task_inputs["attention_mask"].reshape((b_task * k_task, l_task))

        if 'task_labels' in batch:
            task_labels = task_labels.reshape((b_task * k_task, l_task))

        new_w = self.get_new_weights(task="Task")
        input_embeds = F.embedding(task_ids, new_w)

        outputs = self.secondLM.transformer(
            inputs_embeds=input_embeds,
            attention_mask=task_attn,
            output_hidden_states=True
        )

        lm_logits = F.linear(outputs[0], new_w)
        loss_fct = nn.CrossEntropyLoss()

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = task_labels[..., 1:].contiguous()

        lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # #         l = nn.CrossEntropyLoss(reduction="none")

        out_vals = CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
        return out_vals


class MorphMemoryModelMLMOnlineFull(MorphMemoryModel):

    def __init__(self, firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type, buffer):
        super().__init__(firstLM, secondLM, nonces, device, layers, mask_token_id, memory_config, emb_type)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.firstLM.config.hidden_size,
                                                   nhead=self.firstLM.config.num_attention_heads,
                                                   activation='gelu',
                                                   batch_first=True).to(self.device)
        self.emb_gen = nn.TransformerEncoder(encoder_layer, num_layers=1).to(self.device)
        initial_first_ind = int(self.firstLM.config.vocab_size - len(self.nonces))
        initial_second_ind = int(self.secondLM.config.vocab_size - len(self.nonces))

        self.first_list = list(range(initial_first_ind, self.firstLM.config.vocab_size))
        self.second_list = list(range(initial_second_ind, self.secondLM.config.vocab_size))

        self.model_name = "MLMonline_memory_model_{}_{}_{}_memory".format(self.firstLM.config.model_type,
                                                                          self.secondLM.config.model_type,
                                                                          memory_config.agg_method)
        # self.pos_binary = nn.Parameter(torch.randn(1, self.firstLM.config.hidden_size, device=self.device))
        # self.neg_binary = nn.Parameter(torch.randn(1, self.firstLM.config.hidden_size, device=self.device))

        self.cls_token = nn.Parameter(torch.randn(1, self.firstLM.config.hidden_size, device=self.device))

        self.buffer = buffer

    def process_memories(self, mem):

        b, l = mem["input_ids"].shape
        mlm_ids = self.swap_with_mask(mem['input_ids'].to(self.device))
        #         mlm_ids = new_inputs.reshape((b * k, l))
        mlm_attn = mem["attention_mask"].to(self.device)
        first_out = self.firstLM(input_ids=mlm_ids,
                                 attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        for nonce1, nonce2 in zip(self.first_list, self.second_list):
            if nonce1 in mem["input_ids"]:
                #                 msk = (mem["input_ids"] == nonce1)

                #                 embed_ids = msk.nonzero(as_tuple=True)
                #                 if len(combined.shape) == 2:
                #                     nonce_embeds = self.emb_gen(combined[embed_ids[1], :])
                #                 elif len(combined.shape) == 3:
                #                     nonce_embeds = self.emb_gen(combined[embed_ids[0], embed_ids[1], :])
                #                 else:
                #                     raise NotImplementedError("Wrong shape for Combined")
                #                 print(nonce_embeds.shape, "before")
                #                 nonce_embeds = torch.mean(nonce_embeds, dim=0, keepdim=True)
                #                 print(nonce_embeds.shape, "after")
                #                 self.memory.store(nonce2, nonce_embeds)

                src = mem["input_ids"]
                src = src.unsqueeze(-1)
                src = src.expand(-1, -1, self.firstLM.config.hidden_size)

                embed_inputs = combined
                attn = mlm_attn
                attn = torch.cat([torch.tensor([1], device=self.device).unsqueeze(0).expand(attn.shape[0], -1), attn],
                                 dim=1)

                #                 print(cls.shape, embed_inputs.shape)
                embed_inputs = torch.cat([self.cls_token.unsqueeze(0), embed_inputs.unsqueeze(0)], dim=1)
                nonce_embeds = self.emb_gen(embed_inputs, src_key_padding_mask=~attn.bool())
                self.memory.store(nonce2, nonce_embeds[:, 0])

    def swap_with_mask(self, inputs):
        inp = inputs.clone()
        for nonce in self.first_list:
            inp[inp == nonce] = self.mask_token_id
        return inp

    def forward(self, batch):

        mlm_inputs = batch["mlm_inputs"].to(self.device)
        task_inputs = {'input_ids': batch["task_inputs"]['input_ids'].to(self.device),
                       'attention_mask': batch["task_inputs"]['attention_mask'].to(self.device)}
        if 'task_labels' in batch:
            task_labels = batch["task_labels"].to(self.device)

        else:
            task_labels = None

        b, k, l = batch["mlm_inputs"]["input_ids"].shape  # batch x k examples x max_length toks

        new_inputs = self.swap_with_mask(mlm_inputs["input_ids"])  # replace nonce with mask

        mlm_ids = new_inputs.reshape((b * k, l))  # reshape so we have n x seq_len
        mlm_attn = mlm_inputs["attention_mask"].reshape((b * k, l))

        with torch.no_grad():
            first_out = self.firstLM(input_ids=mlm_ids, attention_mask=mlm_attn, output_hidden_states=True)

        first_hidden = first_out.hidden_states

        combined = combine_layers(first_hidden, self.layers)

        chunked = torch.chunk(combined, b)  # get different embeds per nonce, shape = k x max_len x hidden

        # embedding generator + store in memory
        losses = []
        for nonce1, nonce2 in zip(self.first_list, self.second_list):
            if nonce1 in mlm_inputs["input_ids"] and nonce1 not in self.buffer.buffer:
                msk = (mlm_inputs["input_ids"].reshape((b * k, l)) == nonce1)
                src = mlm_inputs["input_ids"].reshape((b * k, l))[msk.nonzero()[:, 0].unique()]
                src = src.unsqueeze(-1)
                src = src.expand(-1, -1, self.firstLM.config.hidden_size)
                #                 pos = self.pos_binary.unsqueeze(0).expand(src.shape[0], -1, -1)
                #                 neg = self.neg_binary.unsqueeze(0).expand(src.shape[0], -1, -1)
                mask = src == nonce1
                #                 added = torch.where(mask, pos, neg)
                cls = self.cls_token.unsqueeze(0).expand(src.shape[0], -1, -1)
                #                 return added, msk
                #                 embed_inputs = combined[msk.nonzero()[:,0].unique()] + added
                embed_inputs = combined[msk.nonzero()[:, 0].unique()]
                attn = mlm_attn[msk.nonzero()[:, 0].unique()]
                attn = torch.cat([torch.tensor([1], device=self.device).unsqueeze(0).expand(attn.shape[0], -1), attn],
                                 dim=1)

                #                 print(src.shape, mask.shape, embed_inputs.shape, self.cls_token.shape)
                embed_inputs = torch.cat([cls, embed_inputs], dim=1)
                nonce_embeds = self.emb_gen(embed_inputs, src_key_padding_mask=~attn.bool())
                self.memory.store(nonce2, nonce_embeds[:, 0])
        #                 return added
        #                 src_mask = torch.einsum("bj, ib->bij", msk.float(), msk.float().T).bool()
        # #                 src_mask = (msk.float().T @ msk.float()).bool()
        #                 print(src_mask)
        #                 print(src_mask.shape)
        #                 print(src_mask.nonzero())
        #                 print(msk.nonzero())
        #                 expanded = src_mask.expand(6, src_mask.shape[0], src_mask.shape[1], src_mask.shape[2])
        #                 print(expanded.shape)
        #                 print(6* src_mask.shape[0], src_mask.shape[1], src_mask.shape[2])
        #                 per_head_mask = expanded.contiguous().view(6* src_mask.shape[0], src_mask.shape[1], src_mask.shape[2])
        # #                 src_mask = torch.ones(msk.shape[1], msk.shape[1], device=self.device)
        # #                 zeros = torch.zeros(msk.shape[1], msk.shape[1], device=self.device)
        # #                 src_mask = torch.where(msk, zeros, torch.ones(1,1, dtype=bool, device=self.device))
        # #                 src_mask[torch.ara]
        # #                 print(src_mask, src_mask.shape, (~src_mask).nonzero())

        #                 embed_ids = msk.nonzero(as_tuple=True)
        #                 print(combined.shape)
        # #                 nonce_embeds = self.emb_gen(combined[embed_ids[0], embed_ids[1], :])
        #                 nonce_embeds = self.emb_gen(combined, mask=~per_head_mask)
        #                 return src_mask, combined, per_head_mask, nonce_embeds
        #                 print(nonce_embeds)
        #                 print(nonce_embeds.shape, "before")
        #                 nonce_embeds = torch.mean(nonce_embeds, dim=0, keepdim=True)
        #                 print(nonce_embeds.shape, "after")
        #                 self.memory.store(nonce2, nonce_embeds)

        # now do task specific stuff
        b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape

        task_ids = task_inputs["input_ids"].reshape((b_task * k_task, l_task))  # reshape so we have n x seq_len
        task_attn = task_inputs["attention_mask"].reshape((b_task * k_task, l_task))

        if 'task_labels' in batch:
            task_labels = task_labels.reshape((b_task * k_task, l_task))

        new_w = self.get_new_weights(task="Task")
        input_embeds = F.embedding(task_ids, new_w)

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
        return out_vals
