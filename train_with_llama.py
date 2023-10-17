import re
from argparse import ArgumentParser

import torch
from datasets import load_from_disk, load_dataset
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataloader import default_collate

from transformers import RobertaForMaskedLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, \
    get_linear_schedule_with_warmup, AdamW, DataCollatorForLanguageModeling, AutoConfig
import accelerate
from accelerate import init_empty_weights, Accelerator
from accelerate import load_checkpoint_and_dispatch
from transformers.modeling_outputs import CausalLMOutputWithPast

from modules.buffer import RetrievalBuffer
from modules.memory import OnlineProtoNet
from modules.model_outputs import CausalLMOutputWithNewToken, CausalLMOutputWithNewTokenNegatives, \
    CausalLMOutputWithRegressionLoss, CausalLMOutputWithRegressionAndNegativeLoss
from modules.utils import combine_layers
from train_utils import get_new_token_loss_labels_llama
import os
from configs.config import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from accelerate import DistributedDataParallelKwargs
import psutil
from modules.aggregators import TransformerSummarizer
import numpy as np


# def get_matching_indices(A, B):
#     used_indices_B = []
#     positions_A_in_B_unique = []
#     for a_item in A:
#         matched_indices = torch.where((B == a_item) & (~torch.tensor([i in used_indices_B for i in range(len(B))], device=B.device)))[0]
#         if len(matched_indices) > 0:
#             positions_A_in_B_unique.append(matched_indices[0].item())
#             used_indices_B.append(matched_indices[0].item())
#         else:
#             positions_A_in_B_unique.append(None)
#
#     used_indices_A = []
#     positions_B_in_A_unique = []
#     for b_item in B:
#         matched_indices = torch.where((A == b_item) & (~torch.tensor([i in used_indices_A for i in range(len(A))], device=A.device)))[0]
#         if len(matched_indices) > 0:
#             positions_B_in_A_unique.append(matched_indices[0].item())
#             used_indices_A.append(matched_indices[0].item())
#         else:
#             positions_B_in_A_unique.append(None)
#
#     ordered_a = order_and_select_indices(positions_A_in_B_unique)
#     ordered_b = order_and_select_indices(positions_B_in_A_unique)
#
#     assert len(ordered_a) == len(ordered_b), "Matching indices must be of the same lengthi, A={}\nB={}".format(ordered_a, ordered_b)
#     return ordered_a, ordered_b
def get_matching_indices(original, modified):
    corresponding_indices = []
    i = j = 0
    while i < len(original) and j < len(modified):
        if original[i] == modified[j]:  # If elements match, append indices to corresponding_indices
            corresponding_indices.append((i, j))
            i += 1
            j += 1
        else:  # If elements do not match, keep incrementing i (index of original) until a match is found or end is reached
            i += 1
            if i == len(original):  # If we reach the end of original and still didn't find a match, increment the modified list index
                i = 0
                j += 1

    indices_in_original = [t[0] for t in corresponding_indices]
    indices_in_nonce = [t[1] for t in corresponding_indices]

    assert len(indices_in_original) == len(indices_in_nonce)
    return indices_in_original, indices_in_nonce



def order_and_select_indices(ind_seq):

    ordered = []
    for item in ind_seq:
        if item is not None:
            if len(ordered) > 0 and item < ordered[-1]:
                ordered.pop()
                ordered.append(item)
            else:
                ordered.append(item)
    return ordered

def decoding_step(logits, temperature, top_k=None, do_sample=False):
    scaled_logits = logits[:, -1, :] / temperature
    if top_k is not None:
        v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
        scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')

    probs = F.softmax(scaled_logits, dim=-1)
    if do_sample:
        idx_next = torch.multinomial(probs, num_samples=1)
    else:
        idx_next = torch.argmax(probs, keepdim=True)

    return idx_next


@torch.no_grad
def generate(model, context, input_ids, attention_mask, max_new_tokens, temperature=1.0, top_k=None, do_sample=False):
    initial_batch = {
        "contexts": [context],
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone()
    }
    initial_outputs = model(initial_batch)

    input_weights = model.get_new_weights(task="Task", memory=initial_outputs.memories[0]['input_memory'])
    output_weights = model.get_new_output_weights(initial_outputs.memories[0]['output_memory'])

    first_token = decoding_step(initial_outputs.logits, temperature, top_k)
    new_input_ids = torch.cat([input_ids, first_token], dim=1)
    last_element = attention_mask[:, -1].unsqueeze(1)
    new_attention_mask = torch.cat([attention_mask, last_element], dim=1)

    for i in range(1, max_new_tokens):
        input_embeds = F.embedding(new_input_ids, input_weights)
        outputs = model.secondLM.model(
            inputs_embeds=input_embeds,
            attention_mask=new_attention_mask
        )
        llama_outputs = model.llama_forward(labels=None, outputs=outputs, new_w=output_weights)

        next_token = decoding_step(llama_outputs.logits, temperature, top_k, do_sample)

        #         print(next_token.shape)
        #         print(new_input_ids.shape)
        new_input_ids = torch.cat([new_input_ids, next_token], dim=1)
        last_element = new_attention_mask[:, -1].unsqueeze(1)
        new_attention_mask = torch.cat([new_attention_mask, last_element], dim=1)

    return new_input_ids


class Memory():
    def __init__(self):
        self.memory = {}

    def store(self, nonce, emb):
        self.memory[nonce] = emb

    def retrieve(self, nonce):
        return self.memory[nonce]

    def __contains__(self, nonce):
        return nonce in self.memory

    def detach(self):
        for k in self.memory:
            self.memory[k].detach()


class EmbeddingGenerator(nn.Module):

    def __init__(self, firstLM, secondLM, num_layers, config):
        super().__init__()
        self.input_hidden_size = firstLM.config.hidden_size
        self.output_hidden_size = secondLM.config.hidden_size
        self.num_attention_heads = firstLM.config.num_attention_heads
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_hidden_size,
                                                   nhead=self.num_attention_heads,
                                                   activation='relu',
                                                   batch_first=True)
        self.num_layers = num_layers
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)

        self.input_emb_head = nn.Linear(self.input_hidden_size, self.output_hidden_size)
        self.output_emb_head = nn.Linear(self.input_hidden_size, self.output_hidden_size)
        self.config = config
        self.agg_method = self.config.agg_method
        if self.agg_method == "CLS":
            input_size = self.config.input_size
            nhead = self.config.nhead
            num_layers = self.config.num_layers
            self.agg = TransformerSummarizer(input_size, nhead, num_layers)



    def forward(self, inputs, attn_mask):

        out = self.encoder(inputs, src_key_padding_mask=~attn_mask.bool())

        out = torch.sum(out * attn_mask.unsqueeze(-1), dim=1) / torch.sum(attn_mask, dim=-1, keepdim=True)

        if self.agg_method == "CLS":
            out = self.agg(out)
        else:
            out = torch.mean(out, dim=0, keepdim=True)

        inp_embeds = self.input_emb_head(out)
        out_embeds = self.output_emb_head(out)

        return inp_embeds, out_embeds



class MorphMemoryModelLLAMA(nn.Module):

    def __init__(self, firstLM, secondLM, num_new_tokens, layers, mask_token_id, memory_config, num_layers, distillation_temp):
        super().__init__()

        self.layers = layers
        self.mask_token_id = mask_token_id
        self.firstLM = firstLM
        self.secondLM = secondLM
        self.memory_config = memory_config
        # self.memory = OnlineProtoNet(memory_config)
        self.num_new_tokens = num_new_tokens
        self.num_layers = num_layers
        self.distillation_temp = distillation_temp

        self.emb_gen = EmbeddingGenerator(self.firstLM, self.secondLM, num_layers, config=self.memory_config)


        self.model_name = "{}_{}".format(self.secondLM.config.model_type, memory_config.agg_method)

        self.dropout = nn.Dropout(0.2)


        # with torch.no_grad():
        #     self.firstLM_mean_embed = torch.mean(self.firstLM.get_input_embeddings().weight[:self.initial_first_ind, :], dim=0)
        #     self.secondLM_mean_embed = torch.mean(self.secondLM.get_input_embeddings().weight[:self.initial_second_ind, :], dim=0)
        #     torch.register_buffer("firstLM_mean_embed", self.firstLM_mean_embed)
        #     torch.register_buffer("secondLM_mean_embed", self.secondLM_mean_embed)


        with torch.no_grad():
            self.firstLM.get_input_embeddings().weight.data[self.first_list, :] = 0.
            self.secondLM.get_input_embeddings().weight[self.second_list, :] = 0.
            self.secondLM.get_output_embeddings().weight[self.second_list] = 0.

        self.freeze()

    @property
    def first_list(self):
        return list(range(self.initial_first_ind, self.firstLM.config.vocab_size))

    @property
    def second_list(self):
        initial_second_ind = int(self.secondLM.config.vocab_size - self.num_new_tokens)
        return list(range(initial_second_ind, self.secondLM.config.vocab_size))

    @property
    def initial_first_ind(self):
        return int(self.firstLM.config.vocab_size - self.num_new_tokens)

    @property
    def initial_second_ind(self):
        return int(self.secondLM.config.vocab_size - self.num_new_tokens)

    def add_new_tokens(self, num_new_tokens):

        self.num_new_tokens += num_new_tokens
        with torch.no_grad():
            self.firstLM.get_input_embeddings().weight[self.first_list, :] = 0.
            self.secondLM.get_input_embeddings().weight[self.second_list, :] = 0.
            self.secondLM.get_output_embeddings().weight[self.second_list] = 0.

    def freeze(self):
        for parameter in self.firstLM.parameters():
            parameter.requires_grad = False

        for parameter in self.secondLM.parameters():
            parameter.requires_grad = False

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    @torch.no_grad
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.secondLM.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.secondLM.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def swap_with_mask(self, inputs):
        inp = inputs.clone()
        for nonce in self.first_list:
            inp[inp == nonce] = self.mask_token_id
        return inp

    def get_new_output_weights(self, memory):
        w = self.secondLM.lm_head.weight.clone()
        n, hidden = w.shape
        # w.requires_grad=True
        msk = torch.zeros_like(w, device=w.device)
        # msk2 = torch.zeros_like(w, device=w.device)
        token_mapping = {k: v for k, v in zip(self.first_list, self.second_list)}
        for key in memory.memory:
            msk = msk.scatter(0, torch.tensor([token_mapping[key]], device=w.device).expand(1, hidden),
                              memory.retrieve(key))
            # msk2[token_mapping[key], :] = 1.

        return w + msk

    def get_new_weights(self, task, memory):

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

        msk = torch.zeros_like(w, device=w.device)
        # msk2 = torch.zeros_like(w, device=w.device)
        token_mapping = {k: v for k, v in zip(self.first_list, self.second_list)}
        for key in memory.memory:
            msk = msk.scatter(0, torch.tensor([token_mapping[key]], device=w.device).expand(1, hidden),
                              memory.retrieve(key))
            # msk2[token_mapping[key], :] = 1.

        return w + msk

    def llama_forward(self, labels, outputs, new_w):
        '''
        Copied from https://github.com/huggingface/transformers/blob/18ee1fe76295239335bf1528c744fe1cfba21cc8/src/transformers/models/llama/modeling_llama.py#L742C7-L742C7
        Note: Output layer weights are not tied to word embedding weights https://github.com/facebookresearch/llama/issues/138
        :param labels:
        :param outputs:
        :return:
        '''
        hidden_states = outputs[0]
        if self.secondLM.config.pretraining_tp > 1:
            lm_head_slices = new_w.split(self.secondLM.vocab_size // self.secondLM.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.secondLM.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = F.linear(hidden_states, new_w, bias=self.secondLM.lm_head.bias)
        logits = logits.float()
        # print(logits.shape, "logits")
        # print(self.secondLM.lm_head.weight.shape, "lm_logits")
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, new_w.shape[0])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(self, batch):
        # nonceMLM = batch["nonceMLM"]
        assert "labels" in batch, "You need labels"
        task_labels = batch["labels"]

        contexts = batch['contexts']

        b_task, l_task = batch["input_ids"].shape

        task_ids = batch['input_ids']
        task_attn = batch["attention_mask"]

        if "negative_input_ids" in batch and "negative_attention_mask" in batch and "negative_labels" in batch:
            negative_ids, negative_attn_mask, negative_labels = batch["negative_input_ids"], batch["negative_attention_mask"], batch['negative_labels']
        else:
            negative_ids, negative_attn_mask, negative_labels = None, None, None

        if "base_input_ids" in batch and "base_attention_mask" in batch and "base_labels" in batch:
            base_ids, base_attn_mask, base_labels = batch["base_input_ids"], batch["base_attention_mask"], batch['base_labels']
        else:
            base_ids, base_attn_mask, base_labels = None, None, None

        # if 'labels' in batch:
        #   task_labels = task_labels.reshape((b_task * k_task, l_task))

        task_labels = batch['labels']
        outs = []
        assert len(contexts) == b_task
        memories = []
        mem_embeds = []
        for i in range(b_task):
            c = contexts[i].to(self.firstLM.device)
            #             print('before', c['input_ids'])
            new_token = c['input_ids'][
                torch.isin(c['input_ids'], torch.tensor(self.first_list, device=c['input_ids'].device))].unique()[
                0].item()

            input_memory = Memory()
            output_memory = Memory()

            mlm_ids = self.swap_with_mask(c['input_ids'])
            #             print('after', c['input_ids'])
            #             print("after mlm ids", mlm_ids)
            with torch.no_grad():
                first_out = self.firstLM(input_ids=mlm_ids, attention_mask=c['attention_mask'],
                                         output_hidden_states=True)

            first_hidden = first_out.hidden_states
            combined = self.dropout(combine_layers(first_hidden, self.layers))

            if len(combined.shape) == 2:
                combined = combined.unsqueeze(0)

            attn = c['attention_mask']
            embed_inputs = combined
            inp_embs, out_embs = self.emb_gen(embed_inputs, attn)

            input_memory.store(new_token, inp_embs)
            output_memory.store(new_token, out_embs)
            new_w = self.get_new_weights(task="Task", memory=input_memory)
            output_weights = self.get_new_output_weights(output_memory)

            input_embeds = F.embedding(task_ids[i], new_w)
            outputs = self.secondLM.model(
                inputs_embeds=input_embeds.unsqueeze(0),
                attention_mask=task_attn[i].unsqueeze(0),
                output_hidden_states=True
            )
            # print(task_labels[i].shape, "label_shape")
            # print(outputs[0].shape)

            llama_outputs = self.llama_forward(task_labels[i], outputs, output_weights)
            #             with torch.no_grad():
            new_tok_loss = get_new_token_loss_labels_llama(task_labels[i].unsqueeze(0), llama_outputs.logits,
                                                         self.secondLM.lm_head.weight.shape[0],
                                                         torch.tensor(self.second_list,
                                                                      device=llama_outputs.logits.device).unique())

            if (negative_ids, negative_attn_mask, negative_labels) != (None, None, None):

                negative_embeds = F.embedding(negative_ids[i], new_w)

                negative_outputs = self.secondLM.model(
                    inputs_embeds=negative_embeds.unsqueeze(0),
                    attention_mask=negative_attn_mask[i].unsqueeze(0),
                    output_hidden_states=True
                )

                negative_llama_outputs = self.llama_forward(negative_labels[i], negative_outputs, output_weights)

                negative_out_vals = CausalLMOutputWithNewTokenNegatives(
                    loss=llama_outputs.loss + negative_llama_outputs.loss,
                    positive_loss=llama_outputs.loss,
                    negative_loss=negative_llama_outputs.loss,
                    positive_logits=llama_outputs.logits,
                    negative_logits=negative_llama_outputs.logits,
                    past_key_values=llama_outputs.past_key_values,
                    hidden_states=llama_outputs.hidden_states,
                    attentions=llama_outputs.attentions,
                    new_token_loss=new_tok_loss,
                    memories=[dict(input_memory=input_memory, output_memory=output_memory)]
                )

            if (base_ids, base_attn_mask, base_labels) != (None, None, None):
                base_outputs = self.secondLM(input_ids=base_ids[i].unsqueeze(0),
                                             attention_mask=base_attn_mask[i].unsqueeze(0),
                                             labels=base_labels[i],
                                             output_hidden_states=True)

                indices_in_base,indices_in_replaced = get_matching_indices(base_ids[i][base_attn_mask[i] == 1].tolist(),
                                                                           task_ids[i][task_attn[i] == 1].tolist())
                # print(indices_in_base, "base")
                # print(indices_in_replaced, "replaced")
                # if self.num_regression_hiddens is None:
                #     cosines = [(1.0-torch.abs(F.cosine_similarity(h1[:, indices_in_replaced], h2[:, indices_in_base], dim=-1))).mean() for h1, h2 in zip(outputs.hidden_states, base_outputs.hidden_states)]
                # else:
                #     cosines = [(1.0-torch.abs(F.cosine_similarity(h1[:, indices_in_replaced], h2[:, indices_in_base], dim=-1))).mean() for h1, h2 in zip(outputs.hidden_states[-self.num_regression_hiddens:], base_outputs.hidden_states[-self.num_regression_hiddens:])]
                cosine_loss = nn.CosineEmbeddingLoss()
                regression_loss = cosine_loss(outputs.hidden_states[-1][:,indices_in_replaced].squeeze(0),
                                              base_outputs.hidden_states[-1][:,indices_in_base].squeeze(0),
                                              target=torch.ones(outputs.hidden_states[-1][:,indices_in_replaced].shape[1], device=base_outputs.hidden_states[-1].device)).mean()

                # cosine_soft = (1.0 - torch.abs(F.cosine_similarity(logsoft_nonce[:, indices_in_replaced, :self.initial_second_ind],
                #                                   logsoft_base[:, indices_in_base, :self.initial_second_ind], dim=-1))).mean()
                soft_base = F.softmax(base_outputs.logits / self.distillation_temp, dim=-1)
                logsoft_nonce = F.log_softmax(llama_outputs.logits / self.distillation_temp, dim=-1)
                distillation_loss = -(soft_base[:, indices_in_base, :self.initial_second_ind] * logsoft_nonce[:, indices_in_replaced, :self.initial_second_ind]).mean()
                distillation_loss = distillation_loss / (self.distillation_temp **2)
                # regression_loss = regression_loss

                # cosines.append(cosine_soft)
                # print(cosines)
                # regression_loss = torch.stack(cosines).mean()

                regression_out_vals = CausalLMOutputWithRegressionLoss(
                    loss=llama_outputs.loss,
                    logits=llama_outputs.logits,
                    base_logits=base_outputs.logits,
                    past_key_values=llama_outputs.past_key_values,
                    hidden_states=llama_outputs.hidden_states,
                    base_hidden_states=base_outputs.hidden_states,
                    attentions=llama_outputs.attentions,
                    new_token_loss=new_tok_loss,
                    memories=[dict(input_memory=input_memory, output_memory=output_memory)],
                    regression_loss=regression_loss,
                    distillation_loss=distillation_loss
                )

            if (negative_ids, negative_attn_mask, negative_labels) != (None, None, None):
                if (base_ids, base_attn_mask, base_labels) != (None, None, None):
                    # a bit hacky way to combine outputs
                    out_vals = CausalLMOutputWithRegressionAndNegativeLoss(
                        loss=negative_out_vals.loss,
                        hidden_states=llama_outputs.hidden_states,
                        positive_loss=negative_out_vals.positive_loss,
                        negative_loss=negative_out_vals.negative_loss,
                        positive_logits=negative_out_vals.positive_logits,
                        negative_logits=negative_out_vals.negative_logits,
                        base_logits=regression_out_vals.base_logits,
                        base_hidden_states=regression_out_vals.base_hidden_states,
                        past_key_values=llama_outputs.past_key_values,
                        attentions=llama_outputs.attentions,
                        new_token_loss=new_tok_loss,
                        memories=[dict(input_memory=input_memory, output_memory=output_memory)],
                        regression_loss=regression_out_vals.regression_loss,
                        distillation_loss=distillation_loss
                    )

                else:
                    out_vals = negative_out_vals

            elif (base_ids, base_attn_mask, base_labels) != (None, None, None):
                out_vals = regression_out_vals



            # print("before mem forward")
            #             print(new_token, new_tok_loss)
            # token_mapping = {k: v for k, v in zip(self.first_list, self.second_list)}
            #             print("output", output_weights[token_mapping[new_token], :])
            #             print("input", new_w[token_mapping[new_token], :])
            else:
                out_vals = CausalLMOutputWithNewToken(
                    loss=llama_outputs.loss,
                    logits=llama_outputs.logits,
                    past_key_values=llama_outputs.past_key_values,
                    hidden_states=llama_outputs.hidden_states,
                    attentions=llama_outputs.attentions,
                    new_token_loss=new_tok_loss,
                    memories=[dict(input_memory=input_memory, output_memory=output_memory)]
                )
            # print("after mem forward")
            outs.append(out_vals)
            # memories.append(memory)

        #         print(outs, "output list")
        final_loss = torch.stack([o.loss for o in outs]).mean()
        final_new_token_loss = [o.new_token_loss for o in outs if o.new_token_loss is not None]
        final_hiddens = [o.hidden_states for o in outs]
        final_past_key_values = [o.past_key_values for o in outs]
        final_attentions = [o.attentions for o in outs]
        if len(final_new_token_loss) > 0:
            final_new_token_loss = torch.stack(final_new_token_loss).mean()
        else:
            final_new_token_loss = None
        #         print([o.new_token_loss for o in outs])
        final_memories = [o.memories[0] for o in outs] # list of the dictionaries

        if (negative_ids, negative_attn_mask, negative_labels) != (None, None, None):
            final_positive_loss = torch.stack([o.positive_loss for o in outs]).mean()
            final_negative_loss = torch.stack([o.negative_loss for o in outs]).mean()
            final_positive_logits = torch.stack([o.positive_logits for o in outs])
            final_negative_logits = torch.stack([o.negative_logits for o in outs])

        if (base_ids, base_attn_mask, base_labels) != (None, None, None):
            final_regression_loss = torch.stack([o.regression_loss for o in outs]).mean()
            final_base_logits = torch.stack([o.base_logits for o in outs])
            final_base_hiddens = [o.base_hidden_states for o in outs]
            final_distillation_loss = torch.stack([o.distillation_loss for o in outs]).mean()

        if (negative_ids, negative_attn_mask, negative_labels) != (None, None, None) and (base_ids, base_attn_mask, base_labels) != (None, None, None):

            return CausalLMOutputWithRegressionAndNegativeLoss(
                loss=final_loss,
                hidden_states=final_hiddens,
                positive_loss=final_positive_loss.detach(),
                negative_loss=final_negative_loss.detach(),
                positive_logits=final_positive_logits,
                negative_logits=final_negative_logits,
                base_logits=final_base_logits,
                base_hidden_states=final_base_hiddens,
                new_token_loss=final_new_token_loss,
                memories=final_memories,
                regression_loss=final_regression_loss,
                distillation_loss=final_distillation_loss
            )
        elif (negative_ids, negative_attn_mask, negative_labels) != (None, None, None):
            return CausalLMOutputWithNewTokenNegatives(
                loss=final_loss,
                positive_loss=final_positive_loss.detach(),
                negative_loss=final_negative_loss.detach(),
                positive_logits=final_positive_logits,
                negative_logits=final_negative_logits,
                hidden_states=final_hiddens,
                attentions=final_attentions,
                new_token_loss=final_new_token_loss,
                memories=final_memories
            )
        elif (base_ids, base_attn_mask, base_labels) != (None, None, None):
            # final_regression_loss = torch.stack([o.regression_loss for o in outs]).mean()
            # final_base_logits = torch.stack([o.base_logits for o in outs])
            final_logits = torch.stack([o.logits for o in outs])
            # final_base_hiddens = [o.base_hidden_states for o in outs]
            return CausalLMOutputWithRegressionLoss(
                loss=final_loss,
                logits=final_logits,
                base_logits=final_base_logits,
                hidden_states=final_hiddens,
                base_hidden_states=final_base_hiddens,
                new_token_loss=final_new_token_loss,
                memories=final_memories,
                regression_loss=final_regression_loss,
                distillation_loss=final_distillation_loss
            )
        else:
            final_logits = torch.cat([o.logits for o in outs], dim=0)
            return CausalLMOutputWithNewToken(
                loss=final_loss,
                logits=final_logits,
                hidden_states=final_hiddens,
                attentions=final_attentions,
                past_key_values=final_past_key_values,
                new_token_loss=final_new_token_loss,
                memories=final_memories
            )


        # print("before return")

        # task_embeds = torch.stack(mem_embeds)
        # outputs = self.secondLM.model(
        #     inputs_embeds=task_embeds,
        #     attention_mask=task_attn,
        #     output_hidden_states=True
        # )
        #
        # return self.secondLM(
        #     inputs_embeds=task_embeds,
        #     attention_mask=task_attn,
        #     labels=task_labels,
        #     output_hidden_states=True
        # )
        # return self.llama_forward(task_labels, outputs)

# class FewShotLlamaDataset(Dataset):
#     def __init__(self, tokenized_dataset, data_collator):
#         self.tokenized_dataset = tokenized_dataset
#         self.data_collator = data_collator
#
#
#     def __getitem__(self, item):
#         tokenized_input = self.tokenized_dataset[item]
#         return tokenized_input
#
#     def collate(self, batch):
#         out_batch = []
#         out_batch.append([b[0] for b in batch])
#
#         out_batch.append(self.data_collator([b[1] for b in batch]))
#
#         return out_batch

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-6)
    # parser.add_argument("--warmup", type=int, default=1e2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--memory", type=str, default="mean")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--word_path", type=str, default='')
    parser.add_argument("--random_ex", action="store_true")
    parser.add_argument("--cat", action="store_true")
    parser.add_argument("--resample", action="store_true")
    parser.add_argument("--prefill", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--negative_examples", action="store_true")
    parser.add_argument("--negative_data_path", type=str, default="")
    parser.add_argument("--regression_objective", action="store_true")
    parser.add_argument("--regression_alpha", type=float, default=1.0)
    parser.add_argument("--distillation_temp", type=int, default=1.0)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--logging_step", type=int, required=True)
    parser.add_argument("--num_eval_steps", type=int, default=1000)
    return parser


def create_checkpoint_directories(args):
    if args.negative_examples and args.regression_objective:
        neg_string = "with_negatives_and_regression"
    elif args.negative_examples:
        neg_string = "with_negatives"
    elif args.regression_objective:
        neg_string = "with_regression"
    else:
        neg_string = "without_negatives_or_regression"


    path = "model_checkpoints/layers/no_mp/llama/input_and_output/filtered/pile/{}_layers/{}_batch_size/{}_agg/{}_examples/lr_{}/weight_decay_{}/{}/"
    path = path.format(args.num_layers, args.batch_size * args.gradient_accumulation_steps,args.memory, args.num_examples, args.lr, args.weight_decay, neg_string)

    if args.negative_examples and args.regression_objective:
        alpha_str = "distillation_weight_{}_temp_{}/".format(args.regression_alpha, args.distillation_temp)
        hidden_str = "output_embedding_cosine/"

        path = path + alpha_str + hidden_str

    suffix = "checkpoints/"
    path = path + suffix
    os.makedirs(path, exist_ok=True)

    return path




def main():
    def tokenize(ex):
        return tokenizerTask(ex['text'], truncation=True, max_length=tokenizerMLM.model_max_length, padding='max_length', return_tensors=None)

    def tokenize_for_buffer(ex):
        return tokenizerMLM(ex['text'], truncation=True, return_tensors="pt")

    def tokenize_regression(ex):

        inps = tokenizerTask(ex['text'], truncation=True, max_length=tokenizerMLM.model_max_length, padding='max_length', return_tensors=None)
        base_inps = tokenizerTask(ex['base text'], truncation=True, max_length=tokenizerMLM.model_max_length, padding='max_length', return_tensors=None)
        row = dict(input_ids=inps['input_ids'],
                attention_mask=inps['attention_mask'],
                base_input_ids=base_inps['input_ids'],
                base_attention_mask=base_inps['attention_mask'])
        return row

    def regression_collate(batch):


        input_batch = [dict(input_ids=b['input_ids'], attention_mask=b['attention_mask']) for b in batch]
        base_batch = [dict(input_ids=b['base_input_ids'], attention_mask=b['base_attention_mask']) for b in batch]

        input_collate = data_collator(input_batch)
        base_collate = data_collator(base_batch)

        base_collate_modified = {"base_" + k:v for k,v in base_collate.items()}
        final_collate = {}
        for coll in [input_collate, base_collate_modified]:
            for k in coll:
                final_collate[k] = coll[k]

        return final_collate


    def check_example(ex):
        found = False
        if re.search(r"\b({})\b".format("|".join(words)), ex['text'], flags=re.I):
            found = True

        return found

    def create_base_and_nonce(ex):
        contained_words = [w for w in words if re.search(r"\b({})\b".format(w), ex['text'], flags=re.I) is not None]

        to_replace = np.random.choice(contained_words)

        original_text =  ex['text']

        split = ex['text'].split(".")
        output = [idx for idx, element in enumerate(split) if re.search(r"\b({})\b".format(to_replace), element, flags=re.I) is not None]

        first_index = output[0]

        new_text = ".".join(split[first_index:])
        # print("replacements and text:")
        # print(to_replace, new_text)

        nonce = "<{}_new>".format(to_replace.lower())

        modified_text = re.sub(r"\b({})\b".format(to_replace), nonce, new_text, flags=re.I)
        # print("modified = {}".format(modified_text))

        ex['base text'] = new_text
        ex['text'] = modified_text

        return ex




    args = get_arguments().parse_args()
    checkpoint_path = create_checkpoint_directories(args)

    # assert not (args.negative_examples and args.regression_objective), "Regression for Negative Examples is not supported"
    assert args.negative_examples == (args.negative_data_path != ""), "There must be a negative data set for negative examples"
    # print("Total Virtual memory usage", dict(psutil.virtual_memory()._asdict()))
    # print("CPU Percent", psutil.cpu_percent())
    # print("Arguments: ", args)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=[ddp_kwargs])
    # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    accelerator.wait_for_everyone()
    tokenizerMLM = AutoTokenizer.from_pretrained("roberta-base", use_fast=False)
    tokenizerTask = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", legacy=True, use_fast=False)
    tokenizerTask.add_bos_token = True
    # tokenizerTask.add_eos_token = True

    tokenizerTask.pad_token = tokenizerTask.unk_token
    word_dict = load_from_disk(args.word_path)


    words = word_dict['train']['words'] + word_dict['test']['words']
    nonces = list(map(lambda w: "<{}_new>".format(w.lower()), words))
    nonces = list(set(nonces))
    # print("Nonces = {}".format(nonces))
    tokenizerMLM.add_tokens(nonces)
    tokenizerTask.add_tokens(nonces)
    mask_token_id = tokenizerMLM.mask_token_id
    print("Total Virtual memory usage", dict(psutil.virtual_memory()._asdict()))
    print("CPU Percent", psutil.cpu_percent())
    token_mapping = {v: k for k, v in zip(tokenizerTask.convert_tokens_to_ids(nonces), tokenizerMLM.convert_tokens_to_ids(nonces))}

    #data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizerTask, return_tensors="pt", padding=True)

    #ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    accelerator.wait_for_everyone()
    firstLM = RobertaForMaskedLM.from_pretrained("roberta-base", low_cpu_mem_usage=True)
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", low_cpu_mem_usage=True)
    print("Total Virtual memory usage", dict(psutil.virtual_memory()._asdict()))
    print("CPU Percent", psutil.cpu_percent())
    # with init_empty_weights():
    #     secondLM = LlamaForCausalLM.from_config(llama_config)

    # firstLM = load_checkpoint_and_dispatch(firstLM, "roberta-large", device_map="auto")
    # secondLM = load_checkpoint_and_dispatch(secondLM, "/vast/work/public/ml-datasets/llama/hf/llama-7b", device_map="auto")

    firstLM.resize_token_embeddings(len(tokenizerMLM))
    secondLM.resize_token_embeddings(len(tokenizerTask)) # pad for speed
    firstLM.eval()
    secondLM.eval()
    print("init memory")
    if args.memory == "mean":
        memory_config = AggregatorConfig()
        # weight_decay = 0.05
    elif args.memory == "cls":
        memory_config = TransformerCLSConfig(
            input_size=firstLM.config.hidden_size,
            nhead=firstLM.config.num_attention_heads,
            num_layers=1
        )

    # elif args.memory == "rnn":
    #     memory_config = RNNAggConfig()
    #     # weight_decay = 0.015
    # elif args.memory == "cls":
    #     memory_config = TransformerCLSConfig()
    else:
        raise NotImplementedError("This memory aggregation is not implemented")
    # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    print("init model")
    accelerator.wait_for_everyone()
    model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), [-1], mask_token_id, memory_config, args.num_layers, args.distillation_temp)
    model = accelerator.prepare(model)
    print("initialized")
    ##pad to multiple of 64
    #for param in firstLM:
     #   param.requires_grad=False
    #for param in secondLM:
     #   param.requires_grad = False

    epochs = args.epochs
    lr = args.lr
    epsilon = 1e-8

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
    print("dataset")
    with accelerator.main_process_first():
        dataset = load_dataset(args.data_path, streaming=True)
        dataset = dataset.filter(check_example)
        dataset = dataset.map(create_base_and_nonce)
        print("tokenizing")
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizerTask, mlm=False, return_tensors="pt")
        if args.regression_objective:
            tokenized_train = dataset['train'].map(tokenize_regression, remove_columns=['text', 'meta', 'base text']).with_format("torch")
            # tokenized_train = tokenized_train.shuffle(buffer_size=10000).with_format("torch")
            train_dl = DataLoader(tokenized_train, batch_size=args.batch_size,
                              collate_fn=regression_collate)

            tokenized_test = dataset['test'].map(tokenize_regression, remove_columns=['text', 'meta', 'base text']).with_format("torch")
            # tokenized_test = tokenized_test.shuffle(buffer_size=2000).with_format("torch")
            test_dl = DataLoader(tokenized_test, batch_size=args.batch_size,
                                 collate_fn=regression_collate)

        else:
            tokenized_train = dataset['train'].map(tokenize, remove_columns=['text', 'meta']).with_format("torch")
            # tokenized_train = tokenized_train.shuffle(buffer_size=10_000).with_format("torch")

            train_dl = DataLoader(tokenized_train, batch_size=args.batch_size,
                              collate_fn=data_collator)

            tokenized_test = dataset['test'].map(tokenize, remove_columns=['text', 'meta']).with_format("torch")

            # tokenized_test = tokenized_test.shuffle(buffer_size=2000).with_format("torch")
            test_dl = DataLoader(tokenized_test, batch_size=args.batch_size,
                                 collate_fn=data_collator)

        buffer = RetrievalBuffer(20, args.num_examples, tokenizerMLM.convert_tokens_to_ids(nonces), tokenizerMLM, tokenizerTask,
                                 args.random_ex, args.cat)
        test_buffer = RetrievalBuffer(20, args.num_examples, tokenizerMLM.convert_tokens_to_ids(nonces), tokenizerMLM, tokenizerTask,
                                      args.random_ex, args.cat)


        if args.negative_examples:
            negative_dataset = load_dataset(args.negative_data_path, streaming=True)
            negative_train_tokenized = negative_dataset['train'].map(tokenize,
                                                                     remove_columns=['text', 'meta']).with_format("torch")
            # negative_train_tokenized = negative_train_tokenized.shuffle(buffer_size=5000).with_format("torch")

            negative_test_tokenized = negative_dataset['test'].map(tokenize,
                                                                   remove_columns=['text', 'meta']).with_format("torch")

            # negative_test_tokenized = negative_test_tokenized.shuffle(buffer_size=5000)
            negative_train_dl = DataLoader(negative_train_tokenized,
                                           batch_size=args.batch_size, collate_fn=data_collator, num_workers=2)
            negative_test_dl = DataLoader(negative_test_tokenized, batch_size=args.batch_size,
                                          collate_fn=data_collator)

    eval_ind = args.logging_step

    opt = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                eps=epsilon,
                lr=lr,
                weight_decay=args.weight_decay
                )

    warmup_steps = int(args.max_steps * 0.03)
    scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, args.max_steps)
    # print("Buffer Nonces = {}".format(buffer.nonces))
    # print("Token Mapping = {}".format(token_mapping))


    # print("loading buffer")
    # tokenized_for_buffer = dataset['train'].map(tokenize_for_buffer, remove_columns=dataset['train'].column_names)
    # buffer_dl = DataLoader(tokenized_for_buffer.with_format('torch'))
    # for inp in buffer_dl:
    #     buffer.store(inp)
    #
    # print("Buffer has {} elements".format(len(buffer.buffer)))
    #
    # test_for_buffer = dataset['test'].map(tokenize_for_buffer, remove_columns=dataset['train'].column_names)
    # buffer_test_dl = DataLoader(test_for_buffer.with_format('torch'))
    # for inp in buffer_test_dl:
    #     test_buffer.store(inp)

    print("Test buffer has {} elements".format(len(test_buffer.buffer)))

    print("Total nonces = {}".format(len(nonces)))
    if args.negative_examples:

        opt, train_dl, test_dl, scheduler, negative_train_dl, negative_test_dl = accelerator.prepare(
            opt, train_dl, test_dl, scheduler, negative_train_dl, negative_test_dl
        )
    else:
        opt, train_dl, test_dl, scheduler = accelerator.prepare(
            opt, train_dl, test_dl, scheduler
        )
    accelerator.register_for_checkpointing(opt)
    accelerator.register_for_checkpointing(scheduler)
    checkpoint_id = 0
    accelerator.init_trackers(
        project_name="fewshot_llama",
        config={"num_examples": args.num_examples,
                "learning_rate": lr,
                "aggregation": memory_config.agg_method,
                "batch_size": args.batch_size,
                "negative examples": args.negative_examples,
                "regression": args.regression_objective,
                "alpha": args.regression_alpha,
                },
    )
    ce_weight = 0.75 # use for weighting the cross entropy, distillation, and regression

    for epoch in range(epochs):
        train_new_token_losses = []
        train_losses = []
        total_loss = 0
        total_new_token_loss = 0
        total_positive_loss = 0
        total_negative_loss = 0
        total_regression_loss = 0
        total_distillation_loss = 0
        for i, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                log_dict = {}

                model.train()
                try:
                    model.module.firstLM.eval()
                    model.module.secondLM.eval()
                except:
                    model.firstLM.eval()
                    model.secondLM.eval()
                # model.zero_grad()

                contexts = []
                for j in range(batch['input_ids'].shape[0]):
                    to_sample = list(set([n for n in buffer.nonces if token_mapping[n] in batch['input_ids'][j]]))
                    assert (len(to_sample) == 1), "Nonces to Sample are {} Should be 1, inputs = {}".format(to_sample, tokenizerTask.decode(batch['input_ids'][j,:]))
                    n = to_sample[0]
                    if n in buffer.buffer:
                        sample = buffer.retrieve(n, batch)
                        if sample is not None:
                            contexts.append(sample)
                        else:
                            print("Null context for {}".format(n))
                    else:
                        seq = tokenizerTask.decode(batch['input_ids'][j,:], skip_special_tokens=True,
                                                clean_up_tokenization_spaces=True)
                        sample = tokenizerMLM([seq],
                                              max_length=tokenizerMLM.model_max_length,
                                              truncation=True,
                                              padding='longest',
                                              return_tensors='pt')
                        contexts.append(sample)

                assert len(contexts) == batch['input_ids'].shape[0], "Context has {} elements when it should have {}".format(len(contexts), batch['input_ids'].shape[0])
                batch['contexts'] = contexts
                if args.negative_examples:
                    neg_train_batch = next(iter(negative_train_dl))
                    batch['negative_input_ids'] = neg_train_batch['input_ids']
                    batch['negative_attention_mask'] = neg_train_batch['attention_mask']
                    batch['negative_labels'] = neg_train_batch['labels']

                # print(batch['input_ids'].shape[0])
                out = model(batch)
                if args.regression_objective and args.negative_examples:
                    distillation_weight = 1.0 - ce_weight - args.regression_alpha
                    loss = out.loss + args.regression_alpha * out.regression_loss + distillation_weight * out.distillation_loss

                elif args.regression_objective:

                    loss = out.regression_loss + out.distillation_loss

                else:
                    loss = out.loss
                # print(loss)

                # train_new_token = accelerator.gather(out.new_token_loss)
                # train_losses.append(loss.item())
                # train_new_token_losses.append(out.new_token_loss.detach().item())
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    for name, param in model.named_parameters():
                        if param.grad is not None and param.requires_grad:
                            log_dict["gradients/post_{}_grad_norm".format(name)] = torch.norm(
                                param.grad.view(-1)).item()
                            if torch.isnan(torch.norm(param.grad.view(-1))):
                                raise Exception("Nan Gradient for {}".format(name))

                opt.step()
                scheduler.step()
                opt.zero_grad()
                model.zero_grad()
                total_loss += loss.detach().float()
                total_new_token_loss += out.new_token_loss.detach().float()
                if args.negative_examples:
                    total_positive_loss += out.positive_loss.detach().float()
                    total_negative_loss += out.negative_loss.detach().float()
                if args.regression_objective:
                    total_regression_loss += out.regression_loss.detach().float()
                    total_distillation_loss += out.distillation_loss.detach().float()





            if accelerator.sync_gradients:
                # accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)

                # for name, param in model.named_parameters():
                #     if param.grad is not None and param.requires_grad:
                #         log_dict["gradients/post_{}_grad_norm".format(name)] = torch.norm(param.grad.view(-1)).item()
                #         if torch.isnan(torch.norm(param.grad.view(-1))):
                #             raise Exception("Nan Gradient for {}".format(name))

                log_dict['train loss'] = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps

                log_dict['train new token loss'] = accelerator.gather(total_new_token_loss).mean().item() / args.gradient_accumulation_steps
                log_dict['num_words_seen'] = len(buffer.buffer)
                total_loss = 0
                total_new_token_loss = 0
                if args.negative_examples:
                    log_dict['train loss on positive examples'] = accelerator.gather(total_positive_loss).mean().item() / args.gradient_accumulation_steps
                    log_dict['train loss on negative examples'] = accelerator.gather(total_negative_loss).mean().item() / args.gradient_accumulation_steps
                    total_negative_loss = 0
                    total_positive_loss = 0

                if args.regression_objective:

                    log_dict['regression loss without weight'] = accelerator.gather(total_regression_loss).mean().item() / args.gradient_accumulation_steps
                    # log_dict['regression loss with alpha'] = (args.regression_alpha * accelerator.gather(total_regression_loss)).mean().item() / args.gradient_accumulation_steps
                    log_dict['distillation loss without weight'] = accelerator.gather(total_distillation_loss).mean().item() / args.gradient_accumulation_steps
                    total_regression_loss = 0
                    total_distillation_loss = 0



                with torch.no_grad():
                    memory_norms = {
                        'input_memory': [],
                        'output_memory': []
                    }

                    for mem_dict in out.memories:
                        for memory_type in ['input_memory', 'output_memory']:
                            m = mem_dict[memory_type]
                            new_ids = list(m.memory.keys())
                            assert len(new_ids) == 1
                            new_id = new_ids[0]
                            memory_norms[memory_type].append(m.retrieve(new_id).norm().detach())

                    for memory_type in ['input_memory', 'output_memory']:
                        norms = memory_norms[memory_type]

                        log_dict["embed_norms/{} token embedding norm".format(memory_type)] = torch.stack(norms).mean().detach().item()

                accelerator.log(log_dict)

                buffer.store_task(batch)
                buffer.cleanup()



            try:
                if i != 0 and i % eval_ind == 0:
                    opt.zero_grad(set_to_none=True)
                    model.eval()
                    with torch.no_grad():
                        total_test_loss = 0
                        total_test_nonce_loss = 0
                        total_test_negative_loss = 0
                        total_test_positive_loss = 0
                        total_test_regression_loss = 0
                        total_test_distillation_loss = 0
                        test_log = {}
                        ct = 0
                        for b in test_dl:
                            if ct >= args.num_eval_steps:
                                break
                            contexts = []
                            for j in range(b['input_ids'].shape[0]):
                                to_sample = list(set([n for n in test_buffer.nonces if token_mapping[n] in b['input_ids'][j]]))
                                assert (len(to_sample) == 1)
                                n = to_sample[0]
                                if n in test_buffer.buffer:
                                    sample = test_buffer.retrieve(n, b)
                                    if sample is not None:
                                        contexts.append(sample)
                                else:
                                    seq = tokenizerTask.decode(b['input_ids'][j,:])
                                    sample = tokenizerMLM([seq],
                                              max_length=tokenizerMLM.model_max_length,
                                              truncation=True,
                                              padding='longest',
                                              return_tensors='pt')
                                    contexts.append(sample)

                            assert len(contexts) == b['input_ids'].shape[0], "Context has {} elements when it should have {}".format(len(contexts), b['input_ids'].shape[0])
                            b['contexts'] = contexts

                            if args.negative_examples:
                                neg_test_batch = next(iter(negative_test_dl))
                                b['negative_input_ids'] = neg_test_batch['input_ids']
                                b['negative_attention_mask'] = neg_test_batch['attention_mask']
                                b['negative_labels'] = neg_test_batch['labels']


                            t_out = model(b)
                            # all_losses = accelerator.gather(t_out.loss)
                            if args.regression_objective and args.negative_examples:
                                distillation_weight = 1.0 - ce_weight - args.regression_alpha
                                total_test_loss += t_out.loss + args.regression_alpha * t_out.regression_loss.detach().float() + distillation_weight * t_out.distillation_loss.detach().float()
                            elif args.regression_objective:
                                total_test_loss += t_out.regression_loss.detach().float() + t_out.distillation_loss.detach().float()
                            else:
                                total_test_loss += t_out.loss.detach().float()
                            try:
                                model.module.memory.memory = {}
                            except:
                                model.memory.memory = {}
                            # all_new_tokens = accelerator.gather(t_out.new_token_loss)
                            total_test_nonce_loss += t_out.new_token_loss.detach()
                            if args.negative_examples:
                                total_test_positive_loss += t_out.positive_loss.detach().float()
                                total_test_negative_loss += t_out.negative_loss.detach().float()

                            if args.regression_objective:
                                total_test_regression_loss += t_out.regression_loss.detach().float()
                                total_test_distillation_loss += t_out.distillation_loss.detach.float()

                            test_buffer.store_task(b)
                            test_buffer.cleanup()


                        avg_test = accelerator.gather(total_test_loss).sum().item() / len(test_dl)
                        avg_new_tok = accelerator.gather(total_test_nonce_loss).sum().item() / len(test_dl)
                        test_log['average test loss'] = avg_test
                        test_log['average test loss on new tokens'] = avg_new_tok
                        test_log['epoch'] = epoch
                        test_log['eval step'] = i // eval_ind

                        if args.negative_examples:
                            test_log['average test loss on positive examples'] = accelerator.gather(total_test_positive_loss).sum().item() / len(test_dl)
                            test_log['average test loss on negative examples'] = accelerator.gather(total_test_negative_loss).sum().item() / len(test_dl)

                        if args.regression_objective:
                            test_log['average regression test loss without alpha'] = accelerator.gather(total_test_regression_loss).sum().item() / len(test_dl)

                        accelerator.log(test_log)
                        accelerator.wait_for_everyone()
                        save_dir = checkpoint_path + "checkpoint_{}_{}".format(epoch, i)
                        os.makedirs(save_dir, exist_ok=True)
                        accelerator.save_state(save_dir)
                        tokenizerMLM.save_pretrained(save_dir + "/tokenizerMLM")
                        tokenizerTask.save_pretrained(save_dir + "tokenizerTask")
                        checkpoint_id += 1

            except:
                accelerator.wait_for_everyone()
                save_dir = checkpoint_path + "checkpoint_{}_{}".format(epoch, i)
                os.makedirs(save_dir, exist_ok=True)
                accelerator.save_state(save_dir)
                tokenizerMLM.save_pretrained(save_dir + "/tokenizerMLM")
                tokenizerTask.save_pretrained(save_dir + "tokenizerTask")
                checkpoint_id += 1

    accelerator.end_training()

if __name__ == "__main__":
    main()

