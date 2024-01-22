import math
import re
from argparse import ArgumentParser
from functools import partial
from math import sqrt
import sys
import torch
from datasets import load_from_disk, load_dataset
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import Dataset, DataLoader, TensorDataset, IterableDataset
# from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe
from torch.utils.data.dataloader import default_collate

from transformers import RobertaForMaskedLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, \
    get_linear_schedule_with_warmup, AdamW, DataCollatorForLanguageModeling, AutoConfig, T5EncoderModel
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
import random
from datetime import datetime, timedelta
import socket
TIME_FORMAT_STR = "%b_%d_%H_%M_%S"

from torch.autograd.profiler import record_function

# os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ['WANDB_DISABLED'] = 'true'
# os.environ['WANDB_MODE']='disabled'

def trace_handler(prof: torch.profiler.profile):
   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"{host_name}_{timestamp}"
   print(prof.key_averages(group_by_stack_n=10).table(sort_by="self_cuda_memory_usage", row_limit=10))

   # Construct the trace file.
   prof.export_chrome_trace(f"{file_prefix}.json.gz")

   # Construct the memory timeline file.
   prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")

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
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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
            if i == len(
                    original):  # If we reach the end of original and still didn't find a match, increment the modified list index
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


def decoding_step(logits, temperature, top_k=None, do_sample=False, new_token_idx = 32000, mask_new_tokens=False):
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
    scaled_logits = logits[:, -1, :] / temperature
    if top_k is not None:
        v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
        scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')

    probs = F.softmax(scaled_logits, dim=-1)
    # print(probs.shape)
    # print("before probs", probs)
    # print("mask tokens is", mask_new_tokens)
    if mask_new_tokens:
        # print("here")
        probs[:,new_token_idx] = 0.0
        # print("after zeros probs", probs)
        # print("sum", probs.sum())
        probs = probs / probs.sum() #renormalize
        # print("after renormalize probs", probs)
    if do_sample:
        idx_next = torch.multinomial(probs, num_samples=1)
    else:
        idx_next = torch.argmax(probs, keepdim=True)

    return idx_next


@torch.no_grad
def generate(model, context, input_ids, attention_mask, max_new_tokens, temperature=1.0, top_k=None, do_sample=False, mask_new_tokens=False):
    initial_batch = {
        "contexts": [context],
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone()
    }
    initial_outputs = model(initial_batch)
    new_tok_id = list(initial_outputs.memories[0]['input_memory'].memory.keys())[0]
    inp_embed = initial_outputs.memories[0]['input_memory'].retrieve(new_tok_id)
    outp_embed = initial_outputs.memories[0]['output_memory'].retrieve(new_tok_id)
    input_weights = model.get_new_weights(task="Task", new_embed=inp_embed)
    output_weights = model.get_new_output_weights(outp_embed)

    first_token = decoding_step(initial_outputs.logits, temperature, top_k, mask_new_tokens=mask_new_tokens)
    new_input_ids = torch.cat([input_ids, first_token], dim=1)
    last_element = attention_mask[:, -1].unsqueeze(1)
    new_attention_mask = torch.cat([attention_mask, last_element], dim=1)
    # print("mask tokens is", mask_new_tokens)
    for i in range(1, max_new_tokens):
        input_embeds = F.embedding(new_input_ids, input_weights)
        outputs = model.secondLM.model(
            inputs_embeds=input_embeds,
            attention_mask=new_attention_mask
        )
        llama_outputs = model.llama_forward(labels=None, outputs=outputs, new_w=output_weights, index=None)

        next_token = decoding_step(llama_outputs.logits, temperature, top_k, do_sample, mask_new_tokens=mask_new_tokens)

        #         print(next_token.shape)
        #         print(new_input_ids.shape)
        new_input_ids = torch.cat([new_input_ids, next_token], dim=1)
        last_element = new_attention_mask[:, -1].unsqueeze(1)
        new_attention_mask = torch.cat([new_attention_mask, last_element], dim=1)

    return new_input_ids

@torch.no_grad
def generate_definition(batch, model, tokenizer):
    
    prompts = ["The word \"<nonce>\" is defined as", "The word \"<nonce>\" means"]
    ctx = batch['contexts']
    outputs = []
    for i in range(batch['input_ids'][0]):
        for prompt in prompts:
            inputs = tokenizer(prompt, truncation=True, return_tensors='pt')
            out = generate(model, ctx[i], inputs['input_ids'][i], inputs['attention_mask'], 50)
            result = {
                'generated definition': tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True),
                'context sentences': [tokenizer.decode(ctx[i][j], 
                                        skip_special_tokens=True, 
                                        clean_up_tokenization_spaces=True) for j in range(ctx[i].shape[0])],
                'prompt': prompt,
            }
            outputs.append(result)
    return outputs

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

class PositionalEncoder(nn.Module):
    """
    self.pe: Positional encoding matrix stored as shape [1, max_len, vector_size]
    """

    def __init__(self, vector_size, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, vector_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, vector_size, 2).float() * (-math.log(10000.0) / vector_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:,:vector_size//2]
        pe = pe.unsqueeze(0)
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        """
        :input x: tensor of shape [batch_size, seq_length, vector_size]
        """
        return x + self.pe[:, :x.size(1)]


class EmbeddingGenerator(nn.Module):

    def __init__(self, firstLM, secondLM, num_layers, config, use_pos=False):
        super().__init__()
        self.input_hidden_size = firstLM.config.hidden_size
        self.output_hidden_size = secondLM.config.hidden_size
        self.num_attention_heads = firstLM.config.num_attention_heads
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_hidden_size,
                                                   nhead=self.num_attention_heads,
                                                   activation='relu',
                                                   batch_first=True)
        self.num_layers = num_layers
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.norm = nn.LayerNorm(self.input_hidden_size)

        self.input_emb_head = nn.Linear(self.input_hidden_size, self.output_hidden_size)
        self.output_emb_head = nn.Linear(self.input_hidden_size, self.output_hidden_size)
        self.config = config
        self.agg_method = self.config.agg_method
        if self.agg_method == "CLS":
            input_size = self.config.input_size
            nhead = self.config.nhead
            num_layers = self.config.num_layers
            self.agg = TransformerSummarizer(input_size, nhead, num_layers)

        if use_pos:
            self.pe = PositionalEncoder(self.input_hidden_size, max_len=512)

        self.use_pos = use_pos


    def calc_init_std(self, desired_mean_norm):
        # calculates the std for initializing the linear year to achieve the desired output norm at initialization
        return sqrt(((desired_mean_norm / sqrt(self.input_hidden_size)) ** 2) / self.output_hidden_size)

    def init_weights(self, input_embed_mean, output_embed_mean):
        input_std = self.calc_init_std(input_embed_mean)
        output_std = self.calc_init_std(output_embed_mean)

        self.input_emb_head.weight.data.normal_(mean=0.0, std=input_std)
        if self.input_emb_head.bias is not None:
            self.input_emb_head.bias.zero_()

        self.output_emb_head.weight.data.normal_(mean=0.0, std=output_std)
        if self.output_emb_head.bias is not None:
            self.output_emb_head.bias.zero_()

    def forward(self, inputs, attn_mask):
        if self.use_pos:
            out = self.pe(inputs)
            out = self.encoder(out, src_key_padding_mask=~attn_mask.bool())
        else:
            out = self.encoder(inputs, src_key_padding_mask=~attn_mask.bool())
        out = self.norm(out)

        out = torch.sum(out * attn_mask.unsqueeze(-1), dim=1) / torch.sum(attn_mask, dim=-1, keepdim=True)

        if self.agg_method == "CLS":
            out = self.agg(out)
        else:
            out = torch.mean(out, dim=0, keepdim=True)

        inp_embeds = self.input_emb_head(out)
        out_embeds = self.output_emb_head(out)

        return inp_embeds, out_embeds


class MorphMemoryModelLLAMA(nn.Module):

    def __init__(self, firstLM, secondLM, num_new_tokens, layers, mask_token_id, memory_config, num_layers,
                 distillation_temp, use_pos=False):
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

        self.emb_gen = EmbeddingGenerator(self.firstLM, self.secondLM, num_layers, config=self.memory_config, use_pos=use_pos)

        self.model_name = "{}_{}".format(self.secondLM.config.model_type, memory_config.agg_method)

        #self.dropout = nn.Dropout(0.2)

        with torch.no_grad():
            # firstLM_mean_embed = torch.mean(self.firstLM.get_output_embeddings().weight[:self.initial_first_ind, :], dim=0)
            output_mean_embed = torch.mean(
                self.secondLM.get_output_embeddings().weight.norm(dim=1))
            # firstLM_std = torch.std(self.firstLM.get_output_embeddings().weight[:self.initial_first_ind, :], dim=0)
            input_mean_embed = torch.mean(
                self.secondLM.get_input_embeddings().weight.norm(dim=1))

            self.emb_gen.init_weights(input_mean_embed, output_mean_embed)

        #     torch.register_buffer("firstLM_mean_embed", self.firstLM_mean_embed)
        #     torch.register_buffer("secondLM_mean_embed", self.secondLM_mean_embed)

        # with torch.no_grad():
        #     self.firstLM.get_input_embeddings().weight.data[self.first_list, :] = 0.
        #     self.secondLM.get_input_embeddings().weight[self.second_list, :] = 0.
        #     self.secondLM.get_output_embeddings().weight[self.second_list] = 0.

        self.freeze()

    @property
    def first_list(self):
        return list(range(self.firstLM.config.vocab_size, self.firstLM.config.vocab_size + self.num_new_tokens))

    @property
    def second_list(self):
        # initial_second_ind = int(self.secondLM.config.vocab_size - self.num_new_tokens)
        return list(range(self.secondLM.config.vocab_size, self.secondLM.config.vocab_size + self.num_new_tokens))

    @property
    def initial_first_ind(self):
        # vocab size + num new tokens - num new tokens
        return self.firstLM.config.vocab_size

    @property
    def initial_second_ind(self):
        return self.secondLM.config.vocab_size

    # def add_new_tokens(self, num_new_tokens):
    #
    #     self.num_new_tokens += num_new_tokens
    #     with torch.no_grad():
    #         self.firstLM.get_input_embeddings().weight[self.first_list, :] = 0.
    #         self.secondLM.get_input_embeddings().weight[self.second_list, :] = 0.
    #         self.secondLM.get_output_embeddings().weight[self.second_list] = 0.

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

    def get_new_output_weights(self, new_embed):
        w = self.secondLM.lm_head.weight
        # n, hidden = w.shape
        # w.requires_grad=True
        # msk = torch.zeros_like(w, device=w.device)
        # # msk2 = torch.zeros_like(w, device=w.device)
        # token_mapping = {k: v for k, v in zip(self.first_list, self.second_list)}
        # for key in memory.memory:
        #     msk = msk.scatter(0, torch.tensor([token_mapping[key]], device=w.device).expand(1, hidden),
        #                       memory.retrieve(key))
        #     # msk2[token_mapping[key], :] = 1.

        # return w + msk
        return torch.cat([w, new_embed])



    def get_new_weights(self, task, new_embed):

        if task == 'MLM':
            ref_model = self.firstLM

        elif task == "Task":
            ref_model = self.secondLM
        else:
            raise NotImplementedError

        w = ref_model.get_input_embeddings().weight
        # n, hidden = w.shape
        # if not ref_model.get_input_embeddings().weight.requires_grad:
        #     w.requires_grad = True
        #
        # msk = torch.zeros_like(w, device=w.device)
        # # msk2 = torch.zeros_like(w, device=w.device)
        # token_mapping = {k: v for k, v in zip(self.first_list, self.second_list)}
        # for key in memory.memory:
        #     msk = msk.scatter(0, torch.tensor([token_mapping[key]], device=w.device).expand(1, hidden),
        #                       memory.retrieve(key))
        #     # msk2[token_mapping[key], :] = 1.
        #
        # return w + msk
        return torch.cat([w, new_embed])

    def get_new_weights_definition_input(self, new_input_embed, def_input_embed):

        input_w = self.secondLM.get_input_embeddings().weight
        # output_w = self.secondLM.get_output_embeddings().weight

        return torch.cat([input_w, new_input_embed, def_input_embed])

    def get_new_weights_definition_output(self, new_output_embed, def_output_embed):
        output_w = self.secondLM.get_output_embeddings().weight
        return torch.cat([output_w, new_output_embed, def_output_embed])

    def llama_forward(self, labels, outputs, new_w, index, new_token_loss=False):
        '''
        Copied from https://github.com/huggingface/transformers/blob/18ee1fe76295239335bf1528c744fe1cfba21cc8/src/transformers/models/llama/modeling_llama.py#L742C7-L742C7
        Note: Output layer weights are not tied to word embedding weights https://github.com/facebookresearch/llama/issues/138
        :param labels:
        :param outputs:
        :return:
        '''
        if index is not None:
            hidden_states = outputs[0][index, :, :]
        else:
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
            if new_token_loss:
                nt_loss_fct = CrossEntropyLoss(reduction='none')
                non_reduced_loss = nt_loss_fct(shift_logits, shift_labels)
                selected = non_reduced_loss[torch.where(torch.isin(torch.flatten(shift_labels), torch.tensor(self.second_list,
                                                                            device=shift_logits.device).unique()))]
                if selected.numel() > 0:
                    new_token_loss = selected.mean()
                else:
                    new_token_loss = None

                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                ), new_token_loss
            else:
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )


        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(self, batch, relational=False):
        # nonceMLM = batch["nonceMLM"]
        # print("batch", batch.keys())
        assert "labels" in batch, "You need labels"
        task_labels = batch["labels"]

        contexts = batch['contexts']

        b_task, l_task = batch["input_ids"].shape

        task_ids = batch['input_ids']
        task_attn = batch["attention_mask"]

        if "negative_input_ids" in batch and "negative_attention_mask" in batch and "negative_labels" in batch:
            negative_ids, negative_attn_mask, negative_labels = batch["negative_input_ids"], batch[
                "negative_attention_mask"], batch['negative_labels']
        else:
            negative_ids, negative_attn_mask, negative_labels = None, None, None

        if "base_input_ids" in batch and "base_attention_mask" in batch and "base_labels" in batch:
            base_ids, base_attn_mask, base_labels = batch["base_input_ids"], batch["base_attention_mask"], batch[
                'base_labels']
        else:
            base_ids, base_attn_mask, base_labels = None, None, None

        # if 'labels' in batch:
        #   task_labels = task_labels.reshape((b_task * k_task, l_task))

        task_labels = batch['labels']
        outs = []
        assert len(contexts) == b_task

        memories = []
        mem_embeds = []
        embeds = []
        neg_embeds = []
        distillation_outputs = []
        definition_outputs = []
        for i in range(b_task):
            # print("Context {}".format(i))
            # with record_function("## MLM STEP ##"):
            c = contexts[i].to(self.firstLM.device)
            #             print('before', c['input_ids'])


            input_memory = Memory()
            output_memory = Memory()

            if self.mask_token_id is not None:
                if not relational:
                    new_token = c['input_ids'][
                        torch.isin(c['input_ids'], torch.tensor(self.first_list, device=c['input_ids'].device))].unique()[
                        0].item()
                    mlm_ids = self.swap_with_mask(c['input_ids'])
                else:
                    new_token = torch.tensor(self.first_list, device=c['input_ids'].device).unique()[0].item()
                    mlm_ids = c['input_ids']
            else:
                new_token = torch.tensor(self.first_list, device=c['input_ids'].device).unique()[0].item()
                mlm_ids = c['input_ids']
            #             print('after', c['input_ids'])
            #             print("after mlm ids", mlm_ids)
            with torch.no_grad():
                first_out = self.firstLM(input_ids=mlm_ids, attention_mask=c['attention_mask'],
                                         output_hidden_states=True)

        # with record_function("## COMBINED ##"):
            first_hidden = first_out.hidden_states
            combined = combine_layers(first_hidden, self.layers)

            if len(combined.shape) == 2:
                combined = combined.unsqueeze(0)

            attn = c['attention_mask']
            embed_inputs = combined

        # with record_function("## EMBED GEN FORWARD ##"):
            inp_embs, out_embs = self.emb_gen(embed_inputs, attn)

            input_memory.store(new_token, inp_embs)
            output_memory.store(new_token, out_embs)

            if "definitions" in batch:
                nonce_def = batch['definitions'][i].to(self.firstLM.device)
                def_inputs = self.swap_with_mask(nonce_def['input_ids'])
                # print("def input ids", def_inputs)
                # pass in a random embed 1/3 of the time
                flag = np.random.binomial(1, 1/3)
                if flag == 1:
                    def_in_embs = torch.randn(1, self.emb_gen.module.output_hidden_size, device=self.firstLM.device)
                    def_out_embs = torch.randn(1, self.emb_gen.module.output_hidden_size, device=self.firstLM.device)
                else:
                    with torch.no_grad():
                        def_out = self.firstLM(input_ids=def_inputs,
                                               attention_mask=nonce_def['attention_mask'],
                                               output_hidden_states=True)
                        def_combined = combine_layers(def_out.hidden_states, self.layers)
                        if len(def_combined.shape) == 2:
                            def_combined = def_combined.unsqueeze(0)

                        def_attn = nonce_def['attention_mask']
                        def_in_embs, def_out_embs = self.emb_gen(def_combined, def_attn)

                definition_outputs.append(dict(input_embed=def_in_embs, output_embed=def_out_embs))
                new_w = self.get_new_weights_definition_input(inp_embs, def_in_embs)
                # print("input weight shape", new_w.shape)
            else:
                new_w = self.get_new_weights(task="Task", new_embed=inp_embs)
            # output_weights = self.get_new_output_weights(new_embed=out_embs)

        # with record_function("## LLAMA MODEL NONCE ##"):
        #     print(task_ids[i])
            input_embeds = F.embedding(task_ids[i], new_w)
            embeds.append(input_embeds)
            mem_embeds.append(dict(input_memory=input_memory, output_memory=output_memory))
            # sys.exit(0)

            # outputs = self.secondLM.model(
            #     inputs_embeds=input_embeds.unsqueeze(0),
            #     attention_mask=task_attn[i].unsqueeze(0),
            #     # output_hidden_states=True
            # )
            # # print(task_labels[i].shape, "label_shape")
            # # print(outputs[0].shape)
            #
            # llama_outputs, new_tok_loss = self.llama_forward(task_labels[i], outputs, output_weights, new_token_loss=True)
        #             with torch.no_grad():
        #     new_tok_loss = get_new_token_loss_labels_llama(task_labels[i].unsqueeze(0), llama_outputs.logits,
        #                                                    self.secondLM.lm_head.weight.shape[0] + self.num_new_tokens,
        #                                                    torch.tensor(self.second_list,
        #                                                                 device=llama_outputs.logits.device).unique())
        # with record_function("## NEGATIVES ##"):
            if (negative_ids, negative_attn_mask, negative_labels) != (None, None, None):
                # print("negative id shape in model", negative_ids[i].shape)
                negative_embeds = F.embedding(negative_ids[i], new_w)
                    # if len(negative_embeds.shape) == 2:
                    #     negative_embeds = negative_embeds.unsqueeze(0)
                    #     n_attn_mask = negative_attn_mask[i].unsqueeze(0)
                    # else:
                    #     n_attn_mask = negative_attn_mask[i]
                neg_embeds.append(negative_embeds)

        if (base_ids, base_attn_mask, base_labels) != (None, None, None):
            with torch.no_grad():
                base_outputs = self.secondLM.model(input_ids=base_ids, attention_mask=base_attn_mask)
                base_final_outs = self.llama_forward(base_labels,
                                                     base_outputs,
                                                     self.secondLM.get_output_embeddings().weight,
                                                     index=None, new_token_loss=False)

        if (negative_ids, negative_attn_mask, negative_labels) != (None, None, None):
            input_embeds = torch.stack(embeds + neg_embeds)
            attn = torch.cat([task_attn, negative_attn_mask], dim=0)
        else:
            input_embeds = torch.stack(embeds)
            attn = task_attn

        outputs = self.secondLM.model(
            inputs_embeds=input_embeds,
            attention_mask=attn,
            # output_hidden_states=True
        )
        loss = []
        new_token_loss = []
        outs = []
        for i, mem in enumerate(mem_embeds):
            out_embs = mem['output_memory'].retrieve(self.firstLM.config.vocab_size)
            if "definitions" in batch:
                output_weights = self.get_new_weights_definition_output(out_embs, definition_outputs[i]['output_embed'])
            else:
                output_weights = self.get_new_output_weights(new_embed=out_embs)
            llama_outputs, new_tok_loss = self.llama_forward(task_labels[i], outputs, output_weights,
                                                             i, new_token_loss=True)
            # loss.append(llama_outputs.loss)
            # new_token_loss.append(new_tok_loss)

            if (negative_ids, negative_attn_mask, negative_labels) != (None, None, None):
                negative_llama_outputs = self.llama_forward(negative_labels[i], outputs, output_weights,
                                                                 i + b_task)

                out_vals = CausalLMOutputWithNewTokenNegatives(
                    loss=llama_outputs.loss + negative_llama_outputs.loss,
                    positive_loss=llama_outputs.loss,
                    negative_loss=negative_llama_outputs.loss,
                    positive_logits=None,
                    negative_logits=None,
                    past_key_values=None,
                    hidden_states=None,
                    attentions=None,
                    new_token_loss=new_tok_loss,
                    memories=[mem]
                )
            if (base_ids, base_attn_mask, base_labels) != (None, None, None):
                indices_in_base, indices_in_replaced = get_matching_indices(
                    base_ids[i][base_attn_mask[i] == 1].tolist(),
                    task_ids[i][task_attn[i] == 1].tolist())

                cosine_loss = nn.CosineEmbeddingLoss()

                regression_loss = cosine_loss(outputs[0][i, indices_in_replaced],
                                              base_outputs[0][i, indices_in_base],
                                              target=torch.ones(
                                                  outputs[0][i, indices_in_replaced].shape[0],
                                                  device=base_outputs[0].device)).mean()

                mse_loss = MSELoss()
                distillation_loss = mse_loss(llama_outputs.logits[indices_in_replaced, :self.initial_second_ind],
                                             base_final_outs.logits[i, indices_in_base, :self.initial_second_ind])

                regression_out_vals = CausalLMOutputWithRegressionLoss(
                    loss=llama_outputs.loss,
                    regression_loss=regression_loss,
                    distillation_loss=distillation_loss,
                    new_token_loss=new_tok_loss,
                    memories=[mem]
                )

            if (negative_ids, negative_attn_mask, negative_labels) != (None, None, None):
                if (base_ids, base_attn_mask, base_labels) != (None, None, None):
                    out_vals = CausalLMOutputWithRegressionAndNegativeLoss(
                        loss=out_vals.loss,
                        hidden_states=None,
                        positive_loss=out_vals.positive_loss,
                        negative_loss=out_vals.negative_loss,
                        positive_logits=None,
                        negative_logits=None,
                        base_logits=None,
                        base_hidden_states=None,
                        past_key_values=None,
                        attentions=None,
                        new_token_loss=new_tok_loss,
                        memories=[mem],
                        regression_loss=regression_loss,
                        distillation_loss=distillation_loss
                    )
            elif (base_ids, base_attn_mask, base_labels) != (None, None, None):
                out_vals = regression_out_vals
            else:
                out_vals = CausalLMOutputWithNewToken(
                    loss=llama_outputs.loss,
                    logits=llama_outputs.logits,
                    past_key_values=None,
                    hidden_states=None,
                    attentions=None,
                    new_token_loss=new_tok_loss,
                    memories=[mem]
                )
            outs.append(out_vals)
        # print("before mem forward")
            #             print(new_token, new_tok_loss)
            # token_mapping = {k: v for k, v in zip(self.first_list, self.second_list)}
            #             print("output", output_weights[token_mapping[new_token], :])
            #             print("input", new_w[token_mapping[new_token], :])
            # else:
            #     out_vals = CausalLMOutputWithNewToken(
            #         loss=llama_outputs.loss,
            #         logits=None,
            #         past_key_values=None,
            #         hidden_states=None,
            #         attentions=None,
            #         new_token_loss=new_tok_loss,
            #         memories=[dict(input_memory=input_memory, output_memory=output_memory)]
            #     )
            # # print("after mem forward")
            # outs.append(out_vals)
            # memories.append(memory)

        #         print(outs, "output list")
        # with record_function("## POST PROCESSING ##"):
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
        final_memories = [o.memories[0] for o in outs]  # list of the dictionaries

        if (negative_ids, negative_attn_mask, negative_labels) != (None, None, None):
            #print("positive losses", torch.stack([o.positive_loss for o in outs]))
            #print("negative losses", torch.stack([o.negative_loss for o in outs]))
            final_positive_loss = torch.stack([o.positive_loss for o in outs]).mean()
            final_negative_loss = torch.stack([o.negative_loss for o in outs]).mean()
            # final_positive_logits = torch.stack([o.positive_logits for o in outs])
            # final_negative_logits = torch.stack([o.negative_logits for o in outs])

        if (base_ids, base_attn_mask, base_labels) != (None, None, None):
            final_regression_loss = torch.stack([o.regression_loss for o in outs]).mean()
            # final_base_logits = torch.stack([o.base_logits for o in outs])
            final_base_hiddens = [o.base_hidden_states for o in outs]
            final_distillation_loss = torch.stack([o.distillation_loss for o in outs]).mean()

        if (negative_ids, negative_attn_mask, negative_labels) != (None, None, None) and (
        base_ids, base_attn_mask, base_labels) != (None, None, None):

            return CausalLMOutputWithRegressionAndNegativeLoss(
                loss=final_loss,
                hidden_states=final_hiddens,
                positive_loss=final_positive_loss.detach(),
                negative_loss=final_negative_loss.detach(),
                positive_logits=None,
                negative_logits=None,
                base_logits=None,
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
                positive_logits=None,
                negative_logits=None,
                hidden_states=final_hiddens,
                attentions=final_attentions,
                new_token_loss=final_new_token_loss,
                memories=final_memories
            )
        elif (base_ids, base_attn_mask, base_labels) != (None, None, None):
            # final_regression_loss = torch.stack([o.regression_loss for o in outs]).mean()
            # final_base_logits = torch.stack([o.base_logits for o in outs])
            # final_logits = torch.stack([o.logits for o in outs])
            # final_base_hiddens = [o.base_hidden_states for o in outs]
            return CausalLMOutputWithRegressionLoss(
                loss=final_loss,
                logits=None,
                base_logits=None,
                hidden_states=final_hiddens,
                base_hidden_states=final_base_hiddens,
                new_token_loss=final_new_token_loss,
                memories=final_memories,
                regression_loss=final_regression_loss,
                distillation_loss=final_distillation_loss
            )
        else:
            final_logits = torch.cat([o.logits for o in outs if o.logits is not None], dim=0)
            return CausalLMOutputWithNewToken(
                loss=final_loss,
                logits=final_logits,
                hidden_states=None,
                attentions=None,
                past_key_values=None,
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


# class ConstantLengthDataset(IterableDataset):
#     """
#     Iterable dataset that returns constant length chunks of tokens from stream of text files.
#         Args:
#             tokenizer (Tokenizer): The processor used for proccessing the data.
#             dataset (dataset.Dataset): Dataset with text files.
#             infinite (bool): If True the iterator is reset after dataset reaches end else stops.
#             seq_length (int): Length of token sequences to return.
#             num_of_sequences (int): Number of token sequences to keep in buffer.
#             chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
#             tokenized (bool): If true we use a pretokenized dataset.
#     """
#
#     def __init__(
#         self,
#         tokenizer,
#         dataset,
#         infinite=False,
#         seq_length=1024,
#         num_of_sequences=1024,
#         chars_per_token=3.6,
#         tokenized=False,
#     ):
#         self.tokenizer = tokenizer
#         self.concat_token_id = tokenizer.bos_token_id
#         self.dataset = dataset
#         self.seq_length = seq_length
#         self.epoch = 0
#         self.infinite = infinite
#         self.current_size = 0
#         self.tokenized = tokenized
#
#         if self.tokenized:
#             self.max_buffer_size = seq_length * num_of_sequences
#             self.content_field = "input_ids"
#         else:
#             self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
#             self.content_field = "content"
#
#     def __iter__(self):
#         iterator = iter(self.dataset)
#         more_examples = True
#         while more_examples:
#             buffer, buffer_len = [], 0
#             while True:
#                 if buffer_len >= self.max_buffer_size:
#                     break
#                 try:
#                     buffer.append(next(iterator)[self.content_field])
#                     buffer_len += len(buffer[-1])
#                 except StopIteration:
#                     if self.infinite:
#                         iterator = iter(self.dataset)
#                         self.epoch += 1
#                         # logger.info(f"Dataset epoch: {self.epoch}")
#                     else:
#                         more_examples = False
#                         break
#             if self.tokenized:
#                 tokenized_inputs = buffer
#             else:
#                 tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
#             all_token_ids = []
#             for tokenized_input in tokenized_inputs:
#                 all_token_ids.extend(tokenized_input + [self.concat_token_id])
#             for i in range(0, len(all_token_ids), self.seq_length):
#                 input_ids = all_token_ids[i : i + self.seq_length]
#                 if len(input_ids) == self.seq_length:
#                     self.current_size += 1
#                     yield torch.tensor(input_ids)
#
#     def shuffle(self, buffer_size=1000):
#         return ShufflerIterDataPipe(self, buffer_size=buffer_size)


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
    # parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--logging_step", type=int, required=True)
    parser.add_argument("--num_eval_steps", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--single_sentence", action="store_true")
    parser.add_argument("--num_feature_layers", type=int, default=1)
    parser.add_argument("--l2", type=float, default=None)
    parser.add_argument("--first_lm", type=str, default="roberta-base")
    parser.add_argument("--progressive_training", action="store_true")
    parser.add_argument("--definition_training", action="store_true")
    parser.add_argument("--use_pos", action="store_true")
    return parser


def create_checkpoint_directories(args):
    if args.negative_examples and args.regression_objective:
        neg_string = "with_negatives_and_regression"
    elif args.negative_examples:
        neg_string = "with_negatives"
    elif args.regression_objective:
        neg_string = "with_regression"
    else:
        neg_string = "vanilla"

    if "interleaved" in args.data_path:
        dataset_name = "interleaved"
    elif "generated" in args.data_path:
        dataset_name = "generated"
    else:
        dataset_name= "pile"

    path = "model_checkpoints/layers/no_mp/llama/input_and_output/filtered/{}/layernorm/{}/{}_layers/last_{}/{}_batch_size/{}_agg/{}_examples/lr_{}/weight_decay_{}/{}/"
    path = path.format(dataset_name, args.first_lm, args.num_layers, args.num_feature_layers,
                       args.batch_size * args.gradient_accumulation_steps * torch.cuda.device_count(), args.memory,
                       args.num_examples, args.lr, args.weight_decay, neg_string)

    if args.regression_objective:
        alpha_str = "distillation_weight_{}_temp_{}/".format(args.regression_alpha, args.distillation_temp)
        hidden_str = "output_embedding_cosine/"

        path = path + alpha_str + hidden_str

    if args.progressive_training:
        path = path + "progressive_training/"
    if args.definition_training:
        path = path + "definition_training/"

    if args.l2 is not None:
        path = path + "l2/"

    if args.use_pos:
        path = path + "positional_encoding/"


    suffix = "checkpoints/"
    # if os.path.isdir(path + suffix):
    #     suffix = "checkpoints2/"
    path = path + suffix
    os.makedirs(path, exist_ok=True)

    return path


def main():
    def prepare_for_t5(seq, nonce):
        t5_format = "<extra_id_{}>"
        ct = 0
        tmp_seq = seq
        while nonce in tmp_seq:
            tmp_seq = tmp_seq.replace(nonce, t5_format.format(ct), 1)
            ct += 1

        return tmp_seq

    def tokenize(ex):
        return tokenizerTask(ex['text'], truncation=True, max_length=256, padding='max_length', return_tensors=None)

    def tokenize_for_buffer(ex):
        return tokenizerMLM(ex['text'], truncation=True, return_tensors="pt")

    def tokenize_regression(ex):
        if args.definition_training:
            text = "<def>" + ex['text']
            inps = tokenizerTask(text, truncation=True, max_length=256, padding='max_length', return_tensors=None)
        else:
            inps = tokenizerTask(ex['text'], truncation=True, max_length=256, padding='max_length', return_tensors=None)
        base_inps = tokenizerTask(ex['base text'], truncation=True, max_length=256, padding='max_length',
                                  return_tensors=None)
        row = dict(input_ids=inps['input_ids'],
                   attention_mask=inps['attention_mask'],
                   base_input_ids=base_inps['input_ids'],
                   base_attention_mask=base_inps['attention_mask'])
        if args.definition_training:
            row['definition'] = ex['definition']
        return row

    def regression_collate(max_num_examples, batch, t5=False, k=None):
        # print(batch[0].keys())
        if k is None:
            num_examples = np.random.choice(max_num_examples) + 1
        else:
            num_examples = k
        contexts = [sample_context(num_examples, b, t5=t5) for b in batch]
        if args.definition_training:
            definitions = [tokenizerMLM(b['definition'], max_length=256, truncation=True, return_tensors='pt') for b in batch]

        input_batch = [dict(input_ids=b['input_ids'], attention_mask=b['attention_mask']) for b in batch]
        base_batch = [dict(input_ids=b['base_input_ids'], attention_mask=b['base_attention_mask']) for b in batch]

        input_collate = data_collator(input_batch)
        base_collate = data_collator(base_batch)

        base_collate_modified = {"base_" + k: v for k, v in base_collate.items()}
        final_collate = {}
        for coll in [input_collate, base_collate_modified]:
            for k in coll:
                final_collate[k] = coll[k]

        final_collate['contexts'] = contexts
        if args.definition_training:
            final_collate['definitions'] = definitions

        return final_collate

    def regular_collate(max_num_examples, batch, t5=False, k=None):
        if k is None:
            num_examples = np.random.choice(max_num_examples) + 1
        else:
            num_examples = k
        contexts = [sample_context(num_examples, b, t5=t5) for b in batch]
        input_batch = [dict(input_ids=b['input_ids'], attention_mask=b['attention_mask']) for b in batch]
        #for b  in batch:
          #  print("sequence", tokenizerTask.decode(b['input_ids']))
        input_collate = data_collator(input_batch)
        final_collate = {}
        for k in input_collate:
            final_collate[k] = input_collate[k]

        final_collate['contexts'] = contexts
        return final_collate

    def sample_context(k, ex, t5=False):
        assert len(ex['sentences']) >= k
        sentences = np.random.choice(ex['sentences'], size=k, replace=False).tolist()
        #print(sentences)
        if t5:
            sentences = [prepare_for_t5(s, "<nonce>") for s in sentences]

        ctx = tokenizerMLM(sentences, max_length=256,
                                        truncation=True,
                                        padding='longest',
                                        return_tensors='pt')

        return ctx

    def check_example(ex):
        found = False
        if re.search(r"\b({})\b".format("|".join(words)), ex['text'], flags=re.I):
            found = True

        return found

    def create_base_and_nonce(ex):
        # contained_words = [w for w in words if re.search(r"\b({})\b".format(w), ex['text'], flags=re.I) is not None]

        # to_replace = np.random.choice(contained_words)
        for w in words:
            if re.search(r"\b({})\b".format(w), ex['text'], flags=re.I) is not None:
                to_replace = w
                break
        # print("to replace", to_replace)
        original_text = ex['text']

        split = ex['text'].split(".")
        output = [idx for idx, element in enumerate(split) if
                  re.search(r"\b({})\b".format(to_replace), element, flags=re.I) is not None]
        # print("output index", output)
        first_index = output[0]

        new_text = ".".join(split[first_index:])
        # print("replacements and text:")
        # print(to_replace, new_text)

        nonce = "<{}_new>".format(to_replace.lower())

        modified_text = re.sub(r"\b({})\b".format(to_replace), nonce, new_text, flags=re.I)
        # print("modified = {}".format(modified_text))
        # print("modified", modified_text)
        # print("base", new_text)
        ex['base text'] = new_text
        ex['text'] = modified_text

        return ex

    def batched_process(batch):

        new_texts = []
        base_texts = []
        for text in batch['text']:
            if re.search(r"\b({})\b".format("|".join(words)), text, flags=re.I):
                contained_words = [w for w in words if re.search(r"\b({})\b".format(w), text, flags=re.I) is not None]
                to_replace = np.random.choice(contained_words)
                split = text.split(".")
                output = [idx for idx, element in enumerate(split) if
                          re.search(r"\b({})\b".format(to_replace), element, flags=re.I) is not None]
                first_index = output[0]
                new_text = ".".join(split[first_index:])
                nonce = "<{}_new>".format(to_replace.lower())
                modified_text = re.sub(r"\b({})\b".format(to_replace), nonce, new_text, flags=re.I)
                new_texts.append(modified_text)
                base_texts.append(text)

        return {'base text': base_texts, 'text': new_texts}

    def get_examples(nonces, ex):

        new_ex = {}
        for n in nonces:
            if n in ex['text']:
                new_ex['word'] = n
                new_ex['example'] = ex['text']

        return new_ex

    def get_examples_single_sentence(nonces, ex):
        new_ex = {}
        for n in nonces:
            if n in ex['text']:
                sentences = ex['text'].split(".")
                example_sentences = [s + "." for s in sentences if n in s]
                new_ex['word'] = n
                new_ex['example'] = example_sentences

        return new_ex

    def fill_buffer(buffer, ex):
        n = tokenizerMLM.convert_tokens_to_ids(ex['word'])
        if type(ex['example']) == str:
            buffer.buffer[n].appendleft(ex['example'])
        elif type(ex['example']) == list:
            for example in ex['example']:
                example_toks = tokenizerMLM(example, truncation=True, max_length=256, return_tensors='pt')
                if n in example_toks['input_ids']:
                    buffer.buffer[n].appendleft(example)


    g = torch.Generator()
    g.manual_seed(0)
    torch.manual_seed(0)

    args = get_arguments().parse_args()
    checkpoint_path = create_checkpoint_directories(args)

    t5_flag = "t5" in args.first_lm

    # assert not (args.negative_examples and args.regression_objective), "Regression for Negative Examples is not supported"
    if args.negative_examples:
        assert args.negative_data_path != "", "There must be a negative data set for negative examples"

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=args.gradient_accumulation_steps,
                              kwargs_handlers=[ddp_kwargs])

    accelerator.wait_for_everyone()
    if args.resume_from_checkpoint is not None:
        current_checkpoint_path = args.resume_from_checkpoint
        tokenizerMLM = AutoTokenizer.from_pretrained(current_checkpoint_path + "/tokenizerMLM", use_fast=False)
        tokenizerTask = LlamaTokenizer.from_pretrained(current_checkpoint_path + "tokenizerTask",
                                                       legacy=True, use_fast=False)
    if t5_flag:
        tokenizerMLM = AutoTokenizer.from_pretrained(args.first_lm, use_fast=False, legacy=False)
    else:
        tokenizerMLM = AutoTokenizer.from_pretrained(args.first_lm, use_fast=False)

    tokenizerTask = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", legacy=True,
                                                   use_fast=False)
    tokenizerTask.add_bos_token = True
    # tokenizerTask.add_eos_token = True

    tokenizerTask.pad_token = tokenizerTask.unk_token
    if args.word_path != '':
        word_dict = load_from_disk(args.word_path)

        words = word_dict['train']['words'] + word_dict['test']['words']
        nonces = list(map(lambda w: "<{}_new>".format(w.lower()), words))
        nonces = list(set(nonces))
    else:
        nonces = ["<nonce>"]
        if args.definition_training:
            nonces.append("<def>")

        
    # print("Nonces = {}".format(nonces))
    tokenizerMLM.add_tokens(nonces)
    tokenizerTask.add_tokens(nonces)
    if not t5_flag:
        mask_token_id = tokenizerMLM.mask_token_id
    else:
        mask_token_id = None
    #accelerator.wait_for_everyone()

    # torch.cuda.memory._record_memory_history(
    #     max_entries=200000
    # )
    with accelerator.main_process_first():
        if "t5" in args.first_lm:
            T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
            firstLM = T5EncoderModel.from_pretrained(args.first_lm, low_cpu_mem_usage=True).to(accelerator.device)
        elif "roberta" in args.first_lm:
            firstLM = RobertaForMaskedLM.from_pretrained(args.first_lm, low_cpu_mem_usage=True).to(accelerator.device)
        else:
            raise NotImplementedError("LM {} is not supported".format(args.first_lm))

        secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                low_cpu_mem_usage=True, device_map=accelerator.device)

    firstLM.eval()
    secondLM.eval()
    print("init memory")
    if args.memory == "mean":
        memory_config = AggregatorConfig()
        # weight_decay = 0.05
    elif args.memory == "cls":
        memory_config = TransformerCLSConfig(
            input_size=firstLM.config.hidden_size,
            nhead=2,
            num_layers=1
        )

    # elif args.memory == "rnn":
    #     memory_config = RNNAggConfig()
    #     # weight_decay = 0.015
    # elif args.memory == "cls":
    #     memory_config = TransformerCLSConfig()
    else:
        raise NotImplementedError("This memory aggregation is not implemented")

    print("init model")
    accelerator.wait_for_everyone()
    layers = [-1 * (x + 1) for x in range(args.num_feature_layers)]
    model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), layers, mask_token_id, memory_config, args.num_layers,
                                  args.distillation_temp, use_pos=args.use_pos).to(accelerator.device)
    print("first list", model.first_list)
    print("second list", model.second_list)
    model.emb_gen = accelerator.prepare(model.emb_gen)
    print("initialized")

    epochs = args.epochs
    lr = args.lr
    epsilon = 1e-8

    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.emb_gen.named_parameters() if
                       not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.emb_gen.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    print("dataset")
    # with accelerator.main_process_first():
    dataset = load_from_disk(args.data_path)
    # dataset = dataset.filter(check_example)
    # dataset = dataset.map(create_base_and_nonce, num_proc=2)
    print("tokenizing")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizerTask, mlm=False, return_tensors="pt")
    if args.regression_objective:
        tokenized_train = dataset['train'].map(tokenize_regression,
                                               remove_columns=[name for name in dataset['train'].column_names if name != "sentences"],
                                               num_proc=2).with_format("torch")
        # tokenized_train = tokenized_train.shuffle(buffer_size=10000).with_format("torch")
        if args.progressive_training:
            k = args.num_examples
        else:
            k = None

        train_dl = DataLoader(tokenized_train, batch_size=args.batch_size,
                              collate_fn=partial(regression_collate, args.num_examples, t5=t5_flag, k=k), drop_last=True,
                              shuffle=True, worker_init_fn=seed_worker, pin_memory=True)

        tokenized_test = dataset['test'].map(tokenize_regression,
                                             remove_columns=[name for name in dataset['test'].column_names if name != "sentences"],
                                             num_proc=2).with_format("torch")
        # tokenized_test = tokenized_test.shuffle(buffer_size=2000).with_format("torch")
        test_dl = DataLoader(tokenized_test, batch_size=args.batch_size,
                             collate_fn=partial(regression_collate, args.num_examples, t5=t5_flag, k=None), shuffle=True, drop_last=True,
                             worker_init_fn=seed_worker, pin_memory=True)

    else:
        tokenized_train = dataset['train'].map(tokenize,
                                               remove_columns=[name for name in dataset['train'].column_names if name != "sentences"],
                                               num_proc=2).with_format("torch")
        # tokenized_train = tokenized_train.shuffle(buffer_size=10_000).with_format("torch")
        if args.progressive_training:
            k = args.num_examples
        else:
            k = None
        train_dl = DataLoader(tokenized_train, batch_size=args.batch_size,
                              collate_fn=partial(regular_collate, args.num_examples, t5=t5_flag, k=k),
                              shuffle=True, drop_last=True, worker_init_fn=seed_worker,
                              pin_memory=True)

        tokenized_test = dataset['test'].map(tokenize,
                                             remove_columns=[name for name in dataset['test'].column_names if name != "sentences"],
                                             num_proc=2).with_format("torch")

        # tokenized_test = tokenized_test.shuffle(buffer_size=2000).with_format("torch")
        test_dl = DataLoader(tokenized_test, batch_size=args.batch_size,
                             collate_fn=partial(regular_collate, args.num_examples, t5=t5_flag, k=None),
                             shuffle=True, drop_last=True, worker_init_fn=seed_worker,
                             pin_memory=True)


    if args.negative_examples:
        negative_dataset = load_from_disk(args.negative_data_path)
        if negative_dataset['train'].num_rows > dataset['train'].num_rows:
            negative_train = negative_dataset['train'].select(list(range(dataset['train'].num_rows)))
        else:
            negative_train = negative_dataset['train']
        if negative_dataset['test'].num_rows > dataset['test'].num_rows:
            negative_test = negative_dataset['test'].select(list(range(dataset['test'].num_rows)))
        else:
            negative_test = negative_dataset['test']
        negative_train_tokenized = negative_train.map(tokenize,
                                                                 remove_columns=negative_dataset[
                                                                     'train'].column_names,
                                                                 num_proc=2).with_format("torch")
        # negative_train_tokenized = negative_train_tokenized.shuffle(buffer_size=5000).with_format("torch")

        negative_test_tokenized = negative_test.map(tokenize,
                                                               remove_columns=negative_dataset[
                                                                   'train'].column_names, num_proc=2).with_format(
            "torch")

        # negative_test_tokenized = negative_test_tokenized.shuffle(buffer_size=5000)
        negative_train_dl = DataLoader(negative_train_tokenized,
                                       batch_size=args.batch_size, collate_fn=data_collator, shuffle=True,
                                       drop_last=True,
                                       worker_init_fn=seed_worker, pin_memory=True)
        negative_test_dl = DataLoader(negative_test_tokenized, batch_size=args.batch_size,
                                      collate_fn=data_collator, shuffle=True, drop_last=True,
                                      worker_init_fn=seed_worker, pin_memory=True)

    eval_ind = args.logging_step

    opt = AdamW(optimizer_grouped_parameters,
                # betas=(0.85,0.95),
                eps=epsilon,
                lr=lr,
                weight_decay=args.weight_decay
                )

    warmup_steps = int(args.epochs * (len(train_dl) / args.gradient_accumulation_steps) * 0.03)
    scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, args.epochs * len(train_dl))

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
    accelerator.wait_for_everyone()
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
    # if args.regression_objective and args.negative_examples:
        # use for weighting the cross entropy, distillation, and regression
        # distillation_weight = args.regression_alpha
        # ce_weight = 1.0 - (args.regression_alpha + distillation_weight)

    global_step = 0


    if args.resume_from_checkpoint is not None:
        matches = re.search(r'checkpoint_(\d+)_(\d+)', args.resume_from_checkpoint)
        num1, num2 = matches.groups()
        base_epoch = int(num1)
        step = int(num2)  # correct for 0 first step
        # assert step % args.gradient_accumulation_steps == 0, "Choose a checkpoint corresponding to a gradient update"
        print("base epoch", base_epoch)
        #todo: implement for second epoch
        if base_epoch != 0:
            curr_global_step = step
            within_batch_step = args.gradient_accumulation_steps * (curr_global_step - ((base_epoch) * len(train_dl)) + 1)
            print("Within batch step {}".format(within_batch_step))
            print("Training DL len = {}".format(len(train_dl)))
            # curr_global_step = (step // (base_epoch * len(train_dl))) // args.gradient_accumulation_steps
            # curr_neg_step = (step // (base_epoch * len(negative_train_dl))) // args.gradient_accumulation_steps
        else:
            curr_global_step = step
            curr_neg_step = step
            within_batch_step = curr_global_step
            # curr_global_step = step // args.gradient_accumulation_steps
            # curr_neg_step = step // args.gradient_accumulation_steps

        active_train_dl = accelerator.skip_first_batches(train_dl, within_batch_step)
        if args.negative_examples:
            active_negative_train_dl = accelerator.skip_first_batches(negative_train_dl, within_batch_step)
        print("curr step", curr_global_step)
        global_step = curr_global_step
        accelerator.load_state(args.resume_from_checkpoint)
    else:
        active_train_dl = train_dl
        base_epoch = 0
        if args.negative_examples:
            active_negative_train_dl = negative_train_dl

    best_test_loss = 10000000
    best_new_token_loss = 10000000
    print("training")


    # with torch.profiler.profile(
    #         activities=[
    #             torch.profiler.ProfilerActivity.CPU,
    #             # torch.profiler.ProfilerActivity.CUDA,
    #         ],
    #         schedule=torch.profiler.schedule(wait=0, warmup=0, active=10, repeat=1),
    #         # record_shapes=True,
    #         # profile_memory=True,
    #         # with_stack=True,
    #         on_trace_ready=trace_handler,
    # ) as prof:

    for epoch in range(base_epoch, epochs):
        print("epoch", epoch)
        if epoch > base_epoch and args.resume_from_checkpoint is not None:
            active_train_dl = train_dl
        if args.progressive_training and epoch > 0:
            k = args.num_examples - epoch
            if args.regression_objective:
                collator = regression_collate
            else:
                collator = regular_collate

            if k <= 0:
                k=None

            train_dl = DataLoader(tokenized_train, batch_size=args.batch_size,
                                  collate_fn=partial(collator, args.num_examples, t5=t5_flag, k=k),
                                  shuffle=True, drop_last=True, worker_init_fn=seed_worker,
                                  pin_memory=True)

            active_train_dl = accelerator.prepare(train_dl)
            if args.resume_from_checkpoint is not None:
                active_train_dl = accelerator.skip_first_batches(active_train_dl, curr_global_step)
        train_new_token_losses = []
        train_losses = []
        total_loss = 0
        total_new_token_loss = 0
        total_positive_loss = 0
        total_negative_loss = 0
        total_regression_loss = 0
        total_distillation_loss = 0
        for i, batch in enumerate(active_train_dl):
            #if global_step==3:
             #   break
            # prof.step()
            # print("Context is {} sentences".format(batch['contexts'][0]['input_ids'].shape))
            # if i == 3:
            #     try:
            #         torch.cuda.memory._dump_snapshot("memsnap3.pickle")
            #     except Exception as e:
            #         print(f"Failed to capture memory snapshot {e}")
            #     torch.cuda.memory._record_memory_history(enabled=None)
            #     #prof.export_memory_timeline(f"memsnap3.html", device="cuda:0")
            #     break

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

                # contexts = []
                # for j in range(batch['input_ids'].shape[0]):
                #     to_sample = list(set([n for n in buffer.nonces if token_mapping[n] in batch['input_ids'][j]]))
                #     # print("base", tokenizerTask.decode(batch['base_input_ids'][j,:]))
                #     # print(batch['input_ids'].shape[0], "shape")
                #     assert (len(to_sample) == 1), "Nonces to Sample are {} Should be 1, inputs = {}".format(to_sample,
                #                                                                                             tokenizerTask.decode(
                #                                                                                                 batch[
                #                                                                                                     'input_ids'][
                #                                                                                                 j, :]))
                #     n = to_sample[0]
                #     if n in buffer.buffer:
                #         sample = buffer.retrieve(n, batch)
                #         if sample is not None:
                #             contexts.append(sample)
                #         else:
                #             print("Null context for {}".format(n))
                    # else:
                    #     seq = tokenizerTask.decode(batch['input_ids'][j,:], skip_special_tokens=True,
                    #                             clean_up_tokenization_spaces=True)
                    #     sample = tokenizerMLM([seq],
                    #                           max_length=tokenizerMLM.model_max_length,
                    #                           truncation=True,
                    #                           padding='longest',
                    #                           return_tensors='pt')
                    #     contexts.append(sample)

                # assert len(contexts) == batch['input_ids'].shape[
                #     0], "Context has {} elements when it should have {}".format(len(contexts),
                #                                                                 batch['input_ids'].shape[0])
                # batch['contexts'] = contexts
                if args.negative_examples:
                    neg_train_batch = next(iter(active_negative_train_dl))
                    # print("negative ids shape out of model", neg_train_batch['input_ids'].shape)
                    batch['negative_input_ids'] = neg_train_batch['input_ids']
                    batch['negative_attention_mask'] = neg_train_batch['attention_mask']
                    batch['negative_labels'] = neg_train_batch['labels']

                # print(batch['input_ids'].shape[0])
                # with record_function("forward + loss"):

                out = model(batch)

                if args.regression_objective:
                    # distillation_weight = 1.0 - ce_weight - args.regression_alpha
                    loss = out.loss + out.regression_loss + out.distillation_loss
                else:
                    loss = out.loss

                if args.l2 is not None:
                    input_l2_losses = []
                    output_l2_losses = []
                    for mem_dict in out.memories:
                        for memory_type in ['input_memory', 'output_memory']:
                            m = mem_dict[memory_type]
                            new_ids = list(m.memory.keys())
                            assert len(new_ids) == 1
                            new_id = new_ids[0]
                            if memory_type == "input_memory":
                                input_l2_losses.append(m.retrieve(new_id).norm())
                            else:
                                output_l2_losses.append(m.retrieve(new_id).norm())

                    input_l2_loss = torch.stack(input_l2_losses).mean()
                    output_l2_loss = torch.stack(output_l2_losses).mean()
                    l2_loss = args.l2 * (input_l2_loss + output_l2_loss)

                    loss = loss + l2_loss

                # print(loss)

                # train_new_token = accelerator.gather(out.new_token_loss)
                # train_losses.append(loss.item())
                # train_new_token_losses.append(out.new_token_loss.detach().item())
                # with record_function("## backward ##"):
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    for name, param in model.named_parameters():
                        if param.grad is not None and param.requires_grad:
                            log_dict["gradients/post_{}_grad_norm".format(name)] = torch.norm(
                                param.grad.view(-1)).item()
                            if torch.isnan(torch.norm(param.grad.view(-1))):
                                raise Exception("Nan Gradient for {}".format(name))
                        if param.requires_grad and param.grad is None:
                            print(name)
                # with record_function("## opt ##"):
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
                global_step += 1
                log_dict['global step'] = global_step
                log_dict['train loss'] = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps

                log_dict['train new token loss'] = accelerator.gather(
                    total_new_token_loss).mean().item() / args.gradient_accumulation_steps
                # log_dict['num_words_seen'] = len(buffer.buffer)
                total_loss = 0
                total_new_token_loss = 0
                if args.negative_examples:
                    log_dict['train loss on positive examples'] = accelerator.gather(
                        total_positive_loss).mean().item() / args.gradient_accumulation_steps
                    log_dict['train loss on negative examples'] = accelerator.gather(
                        total_negative_loss).mean().item() / args.gradient_accumulation_steps
                    total_negative_loss = 0
                    total_positive_loss = 0

                if args.regression_objective:
                    log_dict['regression loss without weight'] = accelerator.gather(
                        total_regression_loss).mean().item() / args.gradient_accumulation_steps
                    # log_dict['regression loss with alpha'] = (args.regression_alpha * accelerator.gather(total_regression_loss)).mean().item() / args.gradient_accumulation_steps
                    log_dict['distillation loss without weight'] = accelerator.gather(
                        total_distillation_loss).mean().item() / args.gradient_accumulation_steps
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

                        log_dict["embed_norms/{} token embedding norm".format(memory_type)] = torch.stack(
                            norms).mean().detach().item()

                accelerator.log(log_dict)
                # print(log_dict)
                # buffer.store_task(batch)
                # buffer.cleanup()

            if (global_step != 0 and global_step % eval_ind == 0 and i % args.gradient_accumulation_steps == 0 and i != 0) \
                    or (i % len(active_train_dl) ==0 and i != 0 and epoch != 0):
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
                        ct += 1
                        if ct >= args.num_eval_steps:
                            break
                        # contexts = []
                        # for j in range(b['input_ids'].shape[0]):
                        #     to_sample = list(
                        #         set([n for n in test_buffer.nonces if token_mapping[n] in b['input_ids'][j]]))
                        #     assert (len(to_sample) == 1)
                        #     n = to_sample[0]
                        #     if n in test_buffer.buffer:
                        #         sample = test_buffer.retrieve(n, b)
                        #         if sample is not None:
                        #             contexts.append(sample)
                        #     # else:
                        #     #     seq = tokenizerTask.decode(b['input_ids'][j,:])
                        #     #     sample = tokenizerMLM([seq],
                        #     #               max_length=tokenizerMLM.model_max_length,
                        #     #               truncation=True,
                        #     #               padding='longest',
                        #     #               return_tensors='pt')
                        #     #     contexts.append(sample)
                        #
                        # assert len(contexts) == b['input_ids'].shape[
                        #     0], "Context has {} elements when it should have {}".format(len(contexts),
                        #                                                                 b['input_ids'].shape[0])
                        # b['contexts'] = contexts

                        if args.negative_examples:
                            neg_test_batch = next(iter(negative_test_dl))
                            b['negative_input_ids'] = neg_test_batch['input_ids']
                            b['negative_attention_mask'] = neg_test_batch['attention_mask']
                            b['negative_labels'] = neg_test_batch['labels']

                        t_out = model(b)
                        # all_losses = accelerator.gather(t_out.loss)
                        if args.regression_objective:
                            # distillation_weight = 1.0 - ce_weight - args.regression_alpha
                            total_test_loss += t_out.loss + t_out.regression_loss.detach().float() + t_out.distillation_loss.detach().float()
                        else:
                            total_test_loss += t_out.loss.detach().float()

                        # all_new_tokens = accelerator.gather(t_out.new_token_loss)
                        total_test_nonce_loss += t_out.new_token_loss.detach()
                        if args.negative_examples:
                            total_test_positive_loss += t_out.positive_loss.detach().float()
                            total_test_negative_loss += t_out.negative_loss.detach().float()

                        if args.regression_objective:
                            total_test_regression_loss += t_out.regression_loss.detach().float()
                            total_test_distillation_loss += t_out.distillation_loss.detach().float()

                        # test_buffer.store_task(b)
                        # test_buffer.cleanup()

                    avg_test = accelerator.gather(total_test_loss).sum().item() / args.num_eval_steps
                    avg_new_tok = accelerator.gather(total_test_nonce_loss).sum().item() / args.num_eval_steps
                    test_log['average test loss'] = avg_test
                    test_log['average test loss on new tokens'] = avg_new_tok
                    test_log['epoch'] = epoch
                    test_log['eval step'] = i // eval_ind

                    if args.negative_examples:
                        test_log['average test loss on positive examples'] = accelerator.gather(
                            total_test_positive_loss).sum().item() / args.num_eval_steps
                        test_log['average test loss on negative examples'] = accelerator.gather(
                            total_test_negative_loss).sum().item() / args.num_eval_steps

                    if args.regression_objective:
                        test_log['average regression test loss'] = accelerator.gather(
                            total_test_regression_loss).sum().item() / args.num_eval_steps
                        test_log['average distillation test loss'] = accelerator.gather(
                            total_test_distillation_loss).sum().item() / args.num_eval_steps

                    accelerator.log(test_log)

                    if avg_test < best_test_loss or avg_new_tok < best_new_token_loss:
                        best_test_loss = avg_test
                        best_new_token_loss = avg_new_tok
                        save_dir = checkpoint_path + "checkpoint_{}_{}".format(epoch, global_step)
                        num_copies = 0
                        tmp_save_dir = save_dir

                        while os.path.isdir(tmp_save_dir):
                            num_copies += 1
                            tmp_save_dir = save_dir + "_v{}".format(num_copies)

                        save_dir = tmp_save_dir
                        os.makedirs(save_dir, exist_ok=True)
                        accelerator.wait_for_everyone()
                        accelerator.save_state(save_dir)
                        tokenizerMLM.save_pretrained(save_dir + "/tokenizerMLM")
                        tokenizerTask.save_pretrained(save_dir + "tokenizerTask")
                        checkpoint_id += 1

    accelerator.end_training()


if __name__ == "__main__":
    main()
