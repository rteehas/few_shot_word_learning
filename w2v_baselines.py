from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
import numpy as np

from modules.model_outputs import CausalLMOutputWithNewToken
from train_with_llama import decoding_step, Memory
# from HiCE.model import HiCE
from HiCE.util import load_training_corpus
from HiCE.util import pad_sequences
from gensim.models import Word2Vec
import sys

sys.path.insert(0, "./HiCE") # to allow loading HiCE model


class HiCEBaseline(nn.Module):
    def __init__(self, hice_path, input_linear_path, output_linear_path, secondLM):
        super().__init__()
        self.hice = torch.load(hice_path)
        self.input_linear = torch.load(input_linear_path)
        self.output_linear = torch.load(output_linear_path)
        self.secondLM = secondLM

    def forward(self, batch):
        contexts = batch['contexts']
        output_embeds = []
        task_ids = batch['input_ids']
        task_attn = batch["attention_mask"]
        task_labels = batch['labels']
        task_input_embeds = []
        mem_embeds = []
        for i in range(len(contexts)):
            input_memory = Memory()
            output_memory = Memory()
            c = contexts[i].to(self.secondLM.device).unsqueeze(0)
            if "character" in batch:
                if len(batch['character'][i].shape) == 1:
                    vocab = batch['character'][i].unsqueeze(0)
                else:
                    vocab = batch['character'][i]


            pred_embed = self.hice(c, vocab)
            tok_input_embed = self.input_linear(pred_embed)
            tok_output_embed = self.output_linear(pred_embed)
            new_w = self.get_new_input_weights(tok_input_embed)
            input_embeds = F.embedding(task_ids[i], new_w)
            task_input_embeds.append(input_embeds)
            output_embeds.append(tok_output_embed)
            new_token = self.secondLM.config.vocab_size
            input_memory.store(new_token, tok_input_embed)
            output_memory.store(new_token, tok_output_embed)
            mem_embeds.append(dict(input_memory=input_memory, output_memory=output_memory))

        input_embeds = torch.stack(task_input_embeds)
        attn = task_attn

        outputs = self.secondLM.model(
            inputs_embeds=input_embeds,
            attention_mask=attn,
            # output_hidden_states=True
        )
        outs = []
        for tok_output in output_embeds:
            output_weights = self.get_new_output_weights(tok_output)
            llama_outputs = self.llama_forward(task_labels[i], outputs, output_weights,
                                                             i, new_token_loss=False)

            out_vals = CausalLMOutputWithNewToken(
                loss=llama_outputs.loss,
                logits=llama_outputs.logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                new_token_loss=None,
                memories=[None]
            )
            outs.append(out_vals)

        final_loss = torch.stack([o.loss for o in outs]).mean()
        final_logits = torch.cat([o.logits for o in outs if o.logits is not None], dim=0)
        return CausalLMOutputWithNewToken(
            loss=final_loss,
            logits=final_logits,
            hidden_states=None,
            attentions=None,
            past_key_values=None,
            new_token_loss=None,
            memories=mem_embeds
        )

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
    def get_new_input_weights(self, new_input_embed):

        input_w = self.secondLM.get_input_embeddings().weight
        # output_w = self.secondLM.get_output_embeddings().weight

        return torch.cat([input_w, new_input_embed])

    def get_new_output_weights(self, new_output_embed):
        output_w = self.secondLM.get_output_embeddings().weight
        return torch.cat([output_w, new_output_embed])


class AdditiveBaseline(nn.Module):
    def __init__(self, dictionary, input_linear_path, output_linear_path, secondLM):
        super().__init__()
        # self.w2v = w2v
        self.dictionary = dictionary
        self.input_linear = torch.load(input_linear_path)
        self.output_linear = torch.load(output_linear_path)
        self.secondLM = secondLM

    def forward(self, batch):
        contexts = batch['contexts']
        output_embeds = []
        task_ids = batch['input_ids']
        task_attn = batch["attention_mask"]
        task_labels = batch['labels']
        task_input_embeds = []
        mem_embeds = []
        for i in range(len(contexts)):
            input_memory = Memory()
            output_memory = Memory()
            c = contexts[i].to(self.secondLM.device).unsqueeze(0)


            pred_embed = get_w2v_ctx_prediction(c, self.dictionary, pad=0)[0]
            print(pred_embed)
            tok_input_embed = self.input_linear(pred_embed)
            tok_output_embed = self.output_linear(pred_embed)
            new_w = self.get_new_input_weights(tok_input_embed)
            input_embeds = F.embedding(task_ids[i], new_w)
            task_input_embeds.append(input_embeds)
            output_embeds.append(tok_output_embed)
            new_token = self.secondLM.config.vocab_size
            input_memory.store(new_token, tok_input_embed)
            output_memory.store(new_token, tok_output_embed)
            mem_embeds.append(dict(input_memory=input_memory, output_memory=output_memory))

        input_embeds = torch.stack(task_input_embeds)
        attn = task_attn

        outputs = self.secondLM.model(
            inputs_embeds=input_embeds,
            attention_mask=attn,
            # output_hidden_states=True
        )
        outs = []
        for tok_output in output_embeds:
            output_weights = self.get_new_output_weights(tok_output)
            llama_outputs = self.llama_forward(task_labels[i], outputs, output_weights,
                                                             i, new_token_loss=False)

            out_vals = CausalLMOutputWithNewToken(
                loss=llama_outputs.loss,
                logits=llama_outputs.logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                new_token_loss=None,
                memories=[None]
            )
            outs.append(out_vals)

        final_loss = torch.stack([o.loss for o in outs]).mean()
        final_logits = torch.cat([o.logits for o in outs if o.logits is not None], dim=0)
        return CausalLMOutputWithNewToken(
            loss=final_loss,
            logits=final_logits,
            hidden_states=None,
            attentions=None,
            past_key_values=None,
            new_token_loss=None,
            memories=mem_embeds
        )

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
    def get_new_input_weights(self, new_input_embed):

        input_w = self.secondLM.get_input_embeddings().weight
        # output_w = self.secondLM.get_output_embeddings().weight

        return torch.cat([input_w, new_input_embed])

    def get_new_output_weights(self, new_output_embed):
        output_w = self.secondLM.get_output_embeddings().weight
        return torch.cat([output_w, new_output_embed])


class Nonce2VecBaseline(nn.Module):
    def __init__(self, n2v, input_linear_path, output_linear_path, secondLM):
        super().__init__()
        self.n2v = n2v
        self.input_linear = torch.load(input_linear_path)
        self.output_linear = torch.load(output_linear_path)
        self.secondLM = secondLM


def make_hice_batch(contexts, true_word, dictionary, maxlen = 24, pad=0):
    _vocab = {v: i + 1 for v, i in zip('abcdefghijklmnopqrstuvwxyz', range(26))}
    lefts, rights = [], []
    nonce = "<nonce>"
    processed_contexts = []
    data = {}
    data['contexts'] = []

    for sent in contexts:
        sent_without_nonce = " ".join([s for s in sent.split() if nonce not in s])
        toks = dictionary.sent2idx(sent_without_nonce)
        data['contexts'].append(toks)
    data['character'] = torch.LongTensor(pad_sequences([[_vocab[v] for v in true_word if v in _vocab]], maxlen=maxlen))
    data['contexts'] = torch.LongTensor(pad_sequences(data['contexts'], maxlen=maxlen, value=pad, padding='post', truncating='post'))
    return data

def make_w2v_batch(contexts, pad=0):
    nonce = "<nonce>"
    data = {}
    data['contexts'] = []

    for sent in contexts:
        sent_without_nonce = " ".join([s for s in sent.split() if nonce not in s])
        toks = dictionary.sent2idx(sent_without_nonce)
        data['contexts'].append(toks)
    return data

def get_w2v_ctx_prediction(oov_cxt, dictionary, pad=0):
    pred = []
    for pp in oov_cxt.reshape(oov_cxt.shape[0], -1):
        tensors = [torch.tensor(dictionary.idx2vec[pi], device="cuda") for pi in pp if pi != pad]
        if len(tensors) > 0:
            pred.append(torch.mean(torch.stack(tensors)))
        else:
            pred.append(torch.zeros_like(torch.tensor(dictionary.idx2vec[10], device="cuda")))
    # try:
    #     pred = [torch.mean(torch.stack([torch.tensor(dictionary.idx2vec[pi], device="cuda") for pi in pp if pi != pad]), dim=0, keepdim=True) for pp in oov_cxt.reshape(oov_cxt.shape[0], -1)]
    # except:
    #     pred = torch.zeros_like(torch.tensor(dictionary.idx2vec[10], device="cuda"))
    return pred

@torch.no_grad
def generate_hice(model, context, vocab, input_ids, attention_mask, max_new_tokens, temperature=1.0, top_k=None, do_sample=False, mask_new_tokens=False):
    initial_batch = {
        "contexts": [context],
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone(),
        'character': [vocab]
    }
    initial_outputs = model(initial_batch)
    new_tok_id = list(initial_outputs.memories[0]['input_memory'].memory.keys())[0]
    inp_embed = initial_outputs.memories[0]['input_memory'].retrieve(new_tok_id)
    outp_embed = initial_outputs.memories[0]['output_memory'].retrieve(new_tok_id)
    input_weights = model.get_new_input_weights(inp_embed)
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
def generate_additive(model, context, input_ids, attention_mask, max_new_tokens, temperature=1.0, top_k=None, do_sample=False, mask_new_tokens=False):
    initial_batch = {
        "contexts": [context],
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone(),
        # 'character': [vocab]
    }
    initial_outputs = model(initial_batch)
    new_tok_id = list(initial_outputs.memories[0]['input_memory'].memory.keys())[0]
    inp_embed = initial_outputs.memories[0]['input_memory'].retrieve(new_tok_id)
    outp_embed = initial_outputs.memories[0]['output_memory'].retrieve(new_tok_id)
    input_weights = model.get_new_input_weights(inp_embed)
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




def load_dictionary(w2v_dir, corpus_dir, maxlen):
    base_w2v = Word2Vec.load(w2v_dir)
    source_train_dataset, source_valid_dataset, dictionary = load_training_corpus(base_w2v, corpus_dir,
                                                                                  maxlen=maxlen,
                                                                                  freq_lbound=16,
                                                                                  freq_ubound=2**16,
                                                                                  cxt_lbound=2)
    return dictionary
