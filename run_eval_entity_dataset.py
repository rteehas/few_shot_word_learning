from llama_eval import *
from train_with_llama import *
from transformers import RobertaForMaskedLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from datasets import load_from_disk
from functools import partial
import json
from argparse import ArgumentParser
import numpy as np
import itertools
import pandas as pd
import re

def get_log_probs(pred, targ, shift=False):
    NULL_TOKEN = 0  # a placeholder used for masked target locations

    pred = pred.clone()
    targ = targ.clone()
    if shift and pred.dim() == 3:  # Dealing with sequences
        pred = pred[:, :-1]  # Remove last prediction in sequence
        targ = targ[:, 1:]  # Shift to align predictions and targets

    mask = targ != -100
    targ[~mask] = NULL_TOKEN  # Can be any valid token, since we'll throw them out
    unmasked_log_probs = pred.log_softmax(-1).gather(
        -1, targ.unsqueeze(-1)).squeeze(-1)

    pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
    correct = pred_ids == targ
    if pred.dim() == 3:
        correct = (pred_ids == targ).all(-1)  # We want to get the whole sequence right
    acc = correct.float().mean()

    n_tokens = mask.float().sum()
    log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
    log_prob_all = unmasked_log_probs * mask.float()
    prob = (unmasked_log_probs.exp() * mask.float()).sum() / n_tokens
    return {
        "acc": acc,
        "log_prob": log_prob,
        "prob": prob,
        "n_tokens": n_tokens,
        "nll": -log_prob,
        "log_prob_all": log_prob_all
    }

def load_json(path):
    with open(path) as f:
        return [json.loads(l.strip()) for l in f]

def get_random_definition():
    data = load_json(POPULAR_ENT_DATA_PATH)
    rand_def = [
        ex['definition'].replace('<extra_id_0>', ex['def_target'][13:-13]) for
        ex in data]
    return rand_def


def dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.to(device)
        elif isinstance(v, dict):
            new_dict[k] = dict_to(v, device)
        else:
            new_dict[k] = v

    return new_dict

def emb_gen_dict_to(d, device):
    new_dict = {}
    for k, v in d.items():
        if k != "contexts":
            if isinstance(v, torch.Tensor):
                new_dict[k] = v.to(device)
            elif isinstance(v, dict):
                new_dict[k] = dict_to(v, device)
            else:
                new_dict[k] = v

    return new_dict

class CustomDataSetClass(Dataset):

    def __init__(
            self,
            data,
            tokenizer,
            input_len,
            target_len,
            input_text="input_text",
            target_text="target_text"
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.input_len = input_len
        self.label_len = target_len
        self.target_text = self.data[target_text]
        self.input_text = self.data[input_text]

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, index):
        input_text = str(self.input_text[index])
        target_text = str(self.target_text[index])

        input_text = ' '.join(input_text.split())
        target_text = ' '.join(target_text.split())

        input_ = self.tokenizer.batch_encode_plus(
            [input_text],
            max_length=self.input_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors='pt'
        )

        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.label_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors='pt'
        )

        input_ids = input_['input_ids'].squeeze()
        input_mask = input_['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'input_ids': input_ids.to(dtype=torch.long),
            'input_mask': input_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }

def prepend_def_gpt(batch_pre, batch_post, model, dataset_name=None):

    # No def
    with torch.no_grad():
        if dataset_name == 'ecbd':
            ex = {}
            ex['input_ids'] = batch_pre["edit_inner"][0]['probe_sentence'][
                'input_ids'][0].unsqueeze(0)
            ex['labels'] = batch_pre["edit_inner"][0]['probe_sentence'][
                'input_ids'][0].unsqueeze(0)  # Dummy label
            ex['attention_mask'] = batch_pre["edit_inner"][0]['probe_sentence'][
                'attention_mask'][0].unsqueeze(0)
        else:
            ex = {}
            ex['input_ids'] = batch_pre["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
            ex['attention_mask'] = batch_pre["edit_inner"][0]['labels'][
                'attention_mask'][0].unsqueeze(0)
            ex['labels'] = batch_pre["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)

        pre_edit_logits = model(**ex).logits

    # Prepend def
    with torch.set_grad_enabled(False):

        if dataset_name == 'ecbd':
            ex = {}
            ex['input_ids'] = batch_post["edit_inner"][0]['probe_sentence'][
                'input_ids'][0].unsqueeze(0)
            ex['labels'] = batch_post["edit_inner"][0]['probe_sentence'][
                'input_ids'][0].unsqueeze(0)  # Dummy label
            ex['attention_mask'] = batch_post["edit_inner"][0][
                'probe_sentence']['attention_mask'][0].unsqueeze(0)

        else:
            ex = {}
            ex['input_ids'] = batch_post["edit_inner"][0]['labels'][
                'input_ids'][
                0].unsqueeze(0)
            ex['attention_mask'] = batch_post["edit_inner"][0]['labels'][
                'attention_mask'][0].unsqueeze(0)
            ex['labels'] = batch_post["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)

        post_edit_logits = model(**ex).logits


    with torch.no_grad():
        n_probe_labels = batch_pre['edit_inner'][0]['labels']['input_ids'].size(
            0)
        pre_edit_dict = []
        post_edit_dict = []

        for i in range(n_probe_labels):
            if dataset_name == 'ecbd':
                pre_label = \
                batch_pre["edit_inner"][0]["probe_sentence"]['input_ids'][
                    i].unsqueeze(0)
                post_label = \
                batch_post["edit_inner"][0]['probe_sentence']['input_ids'][
                    i].unsqueeze(0)
            else:
                pre_label = \
                batch_pre["edit_inner"][0]['labels']['input_ids'][
                    i].unsqueeze(0)
                post_label = \
                batch_post["edit_inner"][0]['labels']['input_ids'][
                    i].unsqueeze(0)

            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, pre_label, shift=True))

            post_edit_dict.append(
                get_log_probs(post_edit_logits, post_label, shift=True))

    post_loc_dict = None
    pre_loc_dict = None

    return pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict, \
           post_loc_dict, pre_loc_dict

def compute_dist_over_labels_t5(tokenizer, edit_dict, labels_str, labels_tsr):
    n_labels = len(edit_dict)
    assert labels_tsr['input_ids'].size(0) == labels_tsr['attention_mask'].size(
        0) == n_labels

    labels = []
    lls = []
    for i in range(n_labels):
        last_idx = labels_tsr['attention_mask'][
                       i].sum().item() - 2  # Ignore special tokens
        # print(labels_tsr['input_ids'][i])
        # print(tokenizer.convert_ids_to_tokens(labels_tsr['input_ids'][i]))
        # print(tokenizer.convert_ids_to_tokens(
        #     labels_tsr['input_ids'][i, 1:last_idx]))
        # print(edit_dict[i]['log_prob_all'])
        # print(last_idx, edit_dict[i]['log_prob_all'][0, 1:last_idx])

        ll = edit_dict[i]['log_prob_all'][0, 1:last_idx]
        ll = ll.sum().item()
        lls.append(ll)
        label = labels_str[i][13:-13]
        labels.append(label)

    return labels, lognormalize(np.array(lls)), lls


def compute_perplexity_t5(tokenizer, logits, label_ids, label_attention_mask):
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fct(logits.view(-1, logits.size(-1)), label_ids.view(-1))

    batch_size = logits.shape[0]
    perp_loss = []
    for i, l in enumerate(loss.view(batch_size, -1)):
        # Remove </s>, <pad>
        n_tokens = label_attention_mask[i].sum() - 2
        # Exclude <extra_id_0>, <extra_id_1>
        perplexity = torch.exp(
            (l * label_attention_mask[i])[1:n_tokens].mean()).item()
        loss_per_token = list(
            zip(tokenizer.convert_ids_to_tokens(
                label_ids[i].cpu().detach().numpy().tolist())[1:n_tokens],
                [float(s) for s in l.cpu().detach().numpy()[1:n_tokens]]
                )
        )
        perp_loss.append((perplexity, loss_per_token))
    return perp_loss


def compute_total_perplexity(all_outputs):
    nll_loss = []
    for output in all_outputs:
        nll_loss.append(np.mean([x[1] for x in output[1]]))
    perplexity = np.exp(np.mean(nll_loss))
    return perplexity


def compute_dist_over_labels_gpt(tokenizer, edit_dict, labels_str, labels_tsr,
                                 left_context_tsr, right_context_tsr):
    n_labels = len(edit_dict)
    assert labels_tsr['input_ids'].size(0) == labels_tsr['attention_mask'].size(
        0) == n_labels

    labels = []
    lls = []
    for i in range(n_labels):
        total_len = \
        (labels_tsr['input_ids'][i] == tokenizer.pad_token_id).nonzero(
            as_tuple=True)[0][0]
        # left and right contexts are the same for all labels
        left_len = \
        (left_context_tsr['input_ids'][0] == tokenizer.pad_token_id).nonzero(
            as_tuple=True)[0]
        right_len = \
        (right_context_tsr['input_ids'][0] == tokenizer.pad_token_id).nonzero(
            as_tuple=True)[0]
        start_loc = left_len
        span_len = total_len - left_len - right_len

        end_loc = start_loc + span_len
        print(labels_tsr['input_ids'][i], "labels")
        print(total_len, left_len, right_len)
        # print(total_len, left_len, right_len)
        # print(tokenizer.convert_ids_to_tokens(labels_tsr['input_ids'][i]))
        # print(tokenizer.convert_ids_to_tokens(labels_tsr['input_ids'][i,
        # start_loc:end_loc]))
        # print(tokenizer.convert_ids_to_tokens(left_context_tsr['input_ids'][0]))
        # print(tokenizer.convert_ids_to_tokens(right_context_tsr['input_ids'][0]))
        # print(edit_dict[i]['log_prob_all'])
        # print(edit_dict[i]['log_prob_all'][0, start_loc-1:end_loc-1])
        # print()

        # assert edit_dict[i]['log_prob_all'].size(1) == labels_tsr['input_ids'][
        #     i].size(0), (edit_dict[i]['log_prob_all'].size(1),
        #                  labels_tsr['input_ids'][i].size(0))
        ll = edit_dict[i]['log_prob_all'][0, start_loc-1:end_loc-1]
        ll = ll.sum().item()
        lls.append(ll)
        label = labels_str[i][13:-13]
        labels.append(label)

    return labels, lognormalize(np.array(lls)), lls


def compute_perplexity_gpt(tokenizer, logits, label_ids, label_attention_mask,
                           labels_tsr, left_context_tsr, right_context_tsr):
    loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = label_ids[..., 1:].contiguous()
    shift_label_attention_mask = label_attention_mask[..., 1:].contiguous()
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))

    batch_size = logits.shape[0]
    perp_loss = []
    for i, l in enumerate(loss.view(batch_size, -1)):

        total_len = \
        (labels_tsr['input_ids'][i] == tokenizer.pad_token_id).nonzero(
            as_tuple=True)[0]
        # left and right contexts are the same for all labels
        left_len = \
        (left_context_tsr['input_ids'][0] == tokenizer.pad_token_id).nonzero(
            as_tuple=True)[0]
        right_len = \
        (right_context_tsr['input_ids'][0] == tokenizer.pad_token_id).nonzero(
            as_tuple=True)[0]
        start_loc = left_len
        span_len = total_len - left_len - right_len

        end_loc = start_loc + span_len

        perplexity = torch.exp(
            (l * shift_label_attention_mask[i])[
            start_loc-1:end_loc-1].mean()).item()

        # print(tokenizer.convert_ids_to_tokens(
        #         label_ids[i].cpu().detach().numpy().tolist()))
        # print(l.cpu().detach().numpy())
        # print(start_loc, end_loc)
        # print(label_ids.size())
        # print(l.size())


        loss_per_token = list(
            zip(tokenizer.convert_ids_to_tokens(
                label_ids[i].cpu().detach().numpy().tolist())[
                start_loc:end_loc],
                [float(s) for s in l.cpu().detach().numpy()[
                                   start_loc-1:end_loc-1]]  # Shift back by 1
                )
        )

        # print(loss_per_token)
        # print()

        if not loss_per_token:
            print(total_len, left_len, right_len, start_loc, end_loc)
            print(tokenizer.convert_ids_to_tokens(labels_tsr['input_ids'][i]))
            print(tokenizer.convert_ids_to_tokens(
                labels_tsr['input_ids'][i, start_loc:end_loc]))
            print(tokenizer.convert_ids_to_tokens(
                left_context_tsr['input_ids'][0]))
            print(tokenizer.convert_ids_to_tokens(
                right_context_tsr['input_ids'][0]))

            print()
            print(loss_per_token)
            print()

        perp_loss.append((perplexity, loss_per_token))
    return perp_loss


def lognormalize(x):
    a = np.logaddexp.reduce(x)
    return np.exp(x - a)


def plot_dist(labels, pre_probs, post_probs):
    df = pd.DataFrame({'label': labels, 'pre': pre_probs, 'post': post_probs})
    ax = df.plot.bar(x='label', y=['pre', 'post'], rot=90,
                     figsize=(2 * np.sqrt(len(labels)), 5))


def aggregate_results(scores):
    pos_count_s = 0
    pos_count_p = 0
    delta_s = []
    delta_p = []
    odds_s = []
    odds_p = []
    for p in scores:
        if isinstance(p, list):
            _, s1, s2, p1, p2 = p[0]
            for p_ in p[1:]:
                s1 = np.logaddexp(s1, p_[1])
                s2 = np.logaddexp(s2, p_[2])
                p1 += p_[3]
                p2 += p_[4]
        else:
            _, s1, s2, p1, p2 = p

        if s2 > s1:
            pos_count_s += 1
        if p2 > p1:
            pos_count_p += 1
        delta_s.append(np.exp(s2) - np.exp(s1))
        delta_p.append(p2 - p1)
        odds_s.append(np.exp(s2) / np.exp(s1))
        odds_p.append(p2 / p1)
    n = len(scores)

    return n, pos_count_s / n, np.mean(delta_s), np.mean(
        odds_s), pos_count_p / n, np.mean(delta_p), np.mean(odds_p)


def compute_top1_accuracy(pred_dist):
    pre_count = 0
    post_count = 0
    for ex in pred_dist:
        scores, label = ex
        if not isinstance(label, list):
            label = [label]

        # pre
        pre_probs = [s[-2] for s in scores]
        pre_id = np.argmax(pre_probs)
        if scores[pre_id][0] in label:
            pre_count += 1

        # post
        post_probs = [s[-1] for s in scores]
        post_id = np.argmax(post_probs)
        if scores[post_id][0] in label:
            post_count += 1

    n = len(pred_dist)

    return pre_count / n, post_count / n, n

def to_tsr_gpt_entity_inference(tokenizer, ex, device, prepend_def=False,
                                prepend_sent=False, random_def=None):
    '''This function supports a single example only (i.e., bsize=1).'''

    definition = [ex['definition']]
    left_context = [ex['left_context']]
    right_context = [ex['right_context']]
    probe_labels = [v['gpt_labels'] for _, v in ex['probe_sentences'].items()]

    if random_def is not None:
        fake_def = random.choice(random_def)
        probe_sentences = [fake_def + ' ' + v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
        left_context_ps = [fake_def + ' ' + v['left_context_ps'] for _, v
                           in ex['probe_sentences'].items()]
        probe_labels = [[fake_def + ' ' + l for l in pl] for pl in
                        probe_labels]

    elif prepend_def and not prepend_sent:
        probe_sentences = [definition[0] + ' ' + v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
        left_context_ps = [definition[0] + ' ' + v['left_context_ps'] for _, v
                           in ex['probe_sentences'].items()]
        probe_labels = [[definition[0] + ' ' + l for l in pl] for pl in
                        probe_labels]
    elif prepend_sent and not prepend_def:
        probe_sentences = [ex['additional_sent'] + ' ' + v['probe_sentence'] for
                           _, v in ex['probe_sentences'].items()]
        left_context_ps = [ex['additional_sent'] + ' ' + v['left_context_ps']
                           for _, v in ex['probe_sentences'].items()]
        probe_labels = [ex['additional_sent'] + ' ' + pl for pl in probe_labels]
    else:
        probe_sentences = [v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
        left_context_ps = [v['left_context_ps'] for _, v in
                           ex['probe_sentences'].items()]



    right_context_ps = [v['right_context_ps'] for _, v in
                        ex['probe_sentences'].items()]

    cleaned_probe_sentences = [ps.strip(' <|endoftext|>') for ps in
                               probe_sentences]

    # _bleu_score = BLEU.compute(predictions=definition,
    #                            references=cleaned_probe_sentences)
    # _bert_score = BERT_SCORE.compute(predictions=definition,
    #                                  references=cleaned_probe_sentences,
    #                                  lang='en',
    #                                  device=device)
    # _bleurt_score = BLEURT.compute(predictions=definition,
    #                                references=cleaned_probe_sentences)
    # _meteor_score = METEOR.compute(predictions=definition,
    #                                references=cleaned_probe_sentences)

    definition_tok = tokenizer(definition, padding=True, return_tensors="pt")
    def_label_tok = tokenizer(definition, padding=True, return_tensors="pt")
    # left_context_tok = tokenizer(left_context, padding=True,
    #                              return_tensors="pt")
    # right_context_tok = tokenizer(right_context, padding=True,
    #                               return_tensors="pt")
    probe_sentences_tok = [
        tokenizer(ps, padding=True, return_tensors="pt").to(device) for
        ps in probe_sentences]
    probe_labels_tok = [
        tokenizer(pl, padding=True, return_tensors="pt").to(device) for
        pl in probe_labels]
    left_context_ps_tok = [
        tokenizer(lc, padding=True, return_tensors="pt").to(device) for
        lc in left_context_ps]
    right_context_ps_tok = [
        tokenizer(rc, padding=True, return_tensors="pt").to(device) for
        rc in right_context_ps]

    edit_inner = [{'probe_sentence': ps} for ps in probe_sentences_tok]
    for i, ps in enumerate(edit_inner):
        ps['labels'] = probe_labels_tok[i]
        ps['left_context_ps'] = left_context_ps_tok[i]
        ps['right_context_ps'] = right_context_ps_tok[i]

    def_ = {**definition_tok}
    def_["labels"] = def_label_tok["input_ids"]

    batch = {
        "edit_inner": edit_inner,  # Edit examples
        "definition": def_,  # Locality
        "cond": None,
        "labels": None,
        # "bleu_score": _bleu_score,
        # "bert_score": _bert_score,
        # "bleurt_score": _bleurt_score,
        # "meteor_score": _meteor_score
    }

    return dict_to(batch, device)

def format_gpt_data(ex, pad_token='<|endoftext|>'):
    context = ex['definition'].split('<extra_id_0>')
    ex['original_def'] = ex['definition']
    assert len(context) == 2, context
    ex['left_context'] = context[0].strip()
    ex['right_context'] = context[1]
    ex['definition'] = ex['definition'].replace('<extra_id_0>', ex['def_target'][13:-13])
    for _, ps in ex['probe_sentences'].items():
        gpt_labels = []
        gpt_labels.append(ps['probe_sentence'].replace('<extra_id_0>', ps['label'][13:-13]) + ' ' + pad_token)
        ps_context = ps['probe_sentence'].split('<extra_id_0>')
        assert len(ps_context) == 2, ps_context
        ps['left_context_ps'] = ps_context[0].strip() + ' ' + pad_token
        ps['right_context_ps'] = ps_context[1] + ' ' + pad_token
        ps['original_ps'] = ps['probe_sentence']
        ps['probe_sentence'] = ps['probe_sentence'].replace('<extra_id_0>', ps['label'][13:-13]) + ' ' + pad_token
        ps['gpt_labels'] = gpt_labels
        ps['answer_str'] = ps['label'][13:-13]
    return ex


def format_gpt2_data_entity_inferences(ex, pad_token='<|endoftext|>'):
    context = ex['definition'].split('<extra_id_0>')
    ex['original_def'] = ex['definition']
    assert len(context) == 2, context
    ex['left_context'] = context[0].strip()
    ex['right_context'] = context[1]
    ex['definition'] = ex['definition'].replace('<extra_id_0>', ex['def_target'][13:-13])
    label = ex['label']
    for _, ps in ex['probe_sentences'].items():
        gpt_labels = []
        for option in ps['labels']:
            gpt_labels.append(ps['probe_sentence'].replace(
                '<extra_id_0>', option[13:-13]) + pad_token)
        ps_context = ps['probe_sentence'].split('<extra_id_0>')
        assert len(ps_context) == 2, ps_context
        ps['left_context_ps'] = ps_context[0].strip() + pad_token
        ps['right_context_ps'] = ps_context[1] + pad_token
        ps['original_ps'] = ps['probe_sentence']
        ps['probe_sentence'] = ps['probe_sentence'].replace('<extra_id_0>', label) + pad_token
        ps['gpt_labels'] = gpt_labels
        ps['answer_str'] = label
    return ex


def format_emb_gen_entity_inferences(ex, pad_token='<|endoftext|>'):
    #extract the entity and then replace with new token
    # process the def into the context later
    context = ex['definition'].split('<extra_id_0>')
    ex['original_def'] = ex['definition']
    assert len(context) == 2, context
    ex['left_context'] = context[0].strip()
    ex['right_context'] = context[1]
    ex['definition'] = ex['definition'].replace('<extra_id_0>', ex['def_target'][13:-13])
    label = ex['label']
    for _, ps in ex['probe_sentences'].items():
        gpt_labels = []
        for option in ps['labels']:
            gpt_labels.append(ps['probe_sentence'].replace(
                '<extra_id_0>', option[13:-13]) + pad_token)
        ps_context = ps['probe_sentence'].split('<extra_id_0>')
        assert len(ps_context) == 2, ps_context
        ps['left_context_ps'] = ps_context[0].strip() + pad_token
        ps['right_context_ps'] = ps_context[1] + pad_token
        ps['original_ps'] = ps['probe_sentence']
        ps['probe_sentence'] = ps['probe_sentence'].replace('<extra_id_0>', label) + pad_token
        ps['gpt_labels'] = gpt_labels
        ps['answer_str'] = label
    return ex

def baseline_main(new_token=False):
    data_file = "entity_knowledge_propagation/data/entity_inferences/disaster_explicit_attribute_independent.json"
    data = load_json(data_file)
    if new_token:
        data = [convert_new_token_example(ex) for ex in data]
        tokenizerTask = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                       legacy=True,
                                                       use_fast=False)
        secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                    low_cpu_mem_usage=True).to("cuda")

        new_nonces = [ex['ent_str'] for ex in data]
        tokenizerTask.add_tokens(new_nonces)
        secondLM.resize_token_embeddings(len(tokenizerTask))
        print(len(tokenizerTask))
    else:
        tokenizerTask = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                       legacy=True,
                                                       use_fast=False)
        secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                    low_cpu_mem_usage=True).to("cuda")

    print(data_file, len(data))

    tokenizerTask.pad_token = tokenizerTask.unk_token
    data = [format_gpt2_data_entity_inferences(ex, pad_token = tokenizerTask.pad_token) for ex in data]
    to_tsr = to_tsr_gpt_entity_inference
    edit_func = prepend_def_gpt
    device = "cuda"
    all_outputs = []
    for i, ex in enumerate(data):
        output = {'ex_id': ex['ex_id']}
        label = ex['label']
        batch = to_tsr(tokenizerTask, ex, device)
        batch_prepended_def = to_tsr(tokenizerTask,
                                     ex,
                                     device,
                                     prepend_def=True,
                                     prepend_sent=False,
                                     random_def=None)
        _, _, \
        pre_edit_dict, post_edit_dict, \
        post_loc_dict, pre_loc_dict = edit_func(
            batch,
            batch_prepended_def,
            secondLM,
            dataset_name=None)

        j = 0
        labels, pre_probs, pre_lls = compute_dist_over_labels_gpt(
            tokenizerTask,
            pre_edit_dict,
            ex['probe_sentences'][f'template_{j}']['labels'],
            batch["edit_inner"][j]['labels'],
            batch["edit_inner"][j]['left_context_ps'],
            batch["edit_inner"][j]['right_context_ps']
        )

        labels, post_probs, post_lls = compute_dist_over_labels_gpt(
            tokenizerTask,
            post_edit_dict,
            ex['probe_sentences'][f'template_{j}']['labels'],
            batch_prepended_def["edit_inner"][j]['labels'],
            batch_prepended_def["edit_inner"][j]['left_context_ps'],
            batch_prepended_def["edit_inner"][j]['right_context_ps']
        )
        result = None
        pred_dist = None
        if label in labels:
            result = [p for p in
                      zip(labels, pre_lls, post_lls, pre_probs, post_probs)
                      if p[0] == label][0]
            pred_dist = [list(zip(labels, pre_lls, post_lls, pre_probs,
                                  post_probs)), label]
        elif isinstance(label, list):
            label_scores = []
            all_scores = []
            for p in zip(labels, pre_lls, post_lls, pre_probs,
                         post_probs):
                all_scores.append(p)
                if p[0] in label:
                    label_scores.append(p)
            result = label_scores
            pred_dist = [all_scores, label]
        else:
            print('-' * 60)
            print('Probe Sentence {}: {}'.format(j,
                                                 ex['probe_sentences'][
                                                     f'template_{j}'][
                                                     'probe_sentence']))
            print('WARNING: Label not found! {}'.format(label))
            print('         Labels {}'.format(labels))
            for p in zip(labels, pre_lls, post_lls, pre_probs,
                         post_probs):
                print(p)

        # if train_params['COMPUTE_SPECIFICITY']:
        #     assert len(results_specificity) == len(data) - 1, \
        #         (len(results_specificity), len(data))

        output['results'] = result
        output['probs'] = pred_dist
        # output['sim_scores'] = {
        #     'bleu_score': bleu_score,
        #     'bert_score': bert_score,
        #     'bleurt_score': bleurt_score,
        #     'meteor_score': meteor_score,
        # }
        # output['specificity'] = results_specificity
        all_outputs.append(output)
        # bar()

    return all_outputs

def aggregate_results(scores):
    pos_count_s = 0
    pos_count_p = 0
    delta_s = []
    delta_p = []
    odds_s = []
    odds_p = []
    for p in scores:
        if isinstance(p, list):
            _, s1, s2, p1, p2 = p[0]
            for p_ in p[1:]:
                s1 = np.logaddexp(s1, p_[1])
                s2 = np.logaddexp(s2, p_[2])
                p1 += p_[3]
                p2 += p_[4]
        else:
            _, s1, s2, p1, p2 = p

        if s2 > s1:
            pos_count_s += 1
        if p2 > p1:
            pos_count_p += 1
        delta_s.append(np.exp(s2) - np.exp(s1))
        delta_p.append(p2 - p1)
        odds_s.append(np.exp(s2) / np.exp(s1))
        odds_p.append(p2 / p1)
    n = len(scores)

    return n, pos_count_s / n, np.mean(delta_s), np.mean(
        odds_s), pos_count_p / n, np.mean(delta_p), np.mean(odds_p)

def compute_top1_accuracy(pred_dist):
    pre_count = 0
    post_count = 0
    for ex in pred_dist:
        scores, label = ex
        if not isinstance(label, list):
            label = [label]

        # pre
        pre_probs = [s[-2] for s in scores]
        pre_id = np.argmax(pre_probs)
        if scores[pre_id][0] in label:
            pre_count += 1

        # post
        post_probs = [s[-1] for s in scores]
        post_id = np.argmax(post_probs)
        if scores[post_id][0] in label:
            post_count += 1

    n = len(pred_dist)

    return pre_count / n, post_count / n, n

def model_main(path):
    device = "cuda"
    data_file = "entity_knowledge_propagation/data/entity_inferences/disaster_explicit_attribute_independent.json"
    data = load_json(data_file)
    data = [convert_new_token_example(ex) for ex in data]
    tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
    tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
    nonces = list(tokenizerTask.get_added_vocab().keys())
    tokenizerMLM.add_tokens(nonces)
    tokenizerTask.add_tokens(nonces)
    firstLM = RobertaForMaskedLM.from_pretrained("roberta-base", low_cpu_mem_usage=True)
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", low_cpu_mem_usage=True)
    firstLM.resize_token_embeddings(len(tokenizerMLM))
    secondLM.resize_token_embeddings(len(tokenizerTask))


    mask_token_id = tokenizerMLM.mask_token_id
    if "mean" in path:
        memory_config = AggregatorConfig()
    elif "cls" in path:
        memory_config = TransformerCLSConfig(
            input_size=firstLM.config.hidden_size,
            nhead=firstLM.config.num_attention_heads,
            num_layers=1
        )
    model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), [-1], mask_token_id, memory_config, 1, None).to(
        device)
    model.load_state_dict(torch.load(path + "/pytorch_model.bin"), strict=False)
    model.device = device
    new_nonces = [ex['ent_str'] for ex in data]
    new_nonces = list(set(new_nonces))
    tokenizerTask.add_tokens(new_nonces)
    tokenizerMLM.add_tokens(new_nonces)
    secondLM.resize_token_embeddings(len(tokenizerTask))
    firstLM.resize_token_embeddings(len(tokenizerMLM))
    firstLM.eval()
    secondLM.eval()
    new_token_num = len(new_nonces)
    model.add_new_tokens(new_token_num)

    print(data_file, len(data))

    tokenizerTask.pad_token = tokenizerTask.unk_token
    data = [format_gpt2_data_entity_inferences(ex, pad_token=tokenizerTask.pad_token) for ex in data]
    to_tsr = to_tsr_emb_gen
    edit_func = emb_gen_model_no_prepend

    all_outputs = []
    for i, ex in enumerate(data):
        output = {'ex_id': ex['ex_id']}
        label = ex['label']
        batch = to_tsr(tokenizerTask, ex, device)
        batch_prepended_def = to_tsr(tokenizerTask,
                                     ex,
                                     device,
                                     prepend_def=True,
                                     prepend_sent=False,
                                     random_def=None)
        _, _, \
        pre_edit_dict, post_edit_dict, \
        post_loc_dict, pre_loc_dict = edit_func(
            batch,
            batch_prepended_def,
            secondLM,
            dataset_name=None)

        j = 0
        labels, pre_probs, pre_lls = compute_dist_over_labels_gpt(
            tokenizerTask,
            pre_edit_dict,
            ex['probe_sentences'][f'template_{j}']['labels'],
            batch["edit_inner"][j]['labels'],
            batch["edit_inner"][j]['left_context_ps'],
            batch["edit_inner"][j]['right_context_ps']
        )

        labels, post_probs, post_lls = compute_dist_over_labels_gpt(
            tokenizerTask,
            post_edit_dict,
            ex['probe_sentences'][f'template_{j}']['labels'],
            batch_prepended_def["edit_inner"][j]['labels'],
            batch_prepended_def["edit_inner"][j]['left_context_ps'],
            batch_prepended_def["edit_inner"][j]['right_context_ps']
        )
        result = None
        pred_dist = None
        if label in labels:
            result = [p for p in
                      zip(labels, pre_lls, post_lls, pre_probs, post_probs)
                      if p[0] == label][0]
            pred_dist = [list(zip(labels, pre_lls, post_lls, pre_probs,
                                  post_probs)), label]
        elif isinstance(label, list):
            label_scores = []
            all_scores = []
            for p in zip(labels, pre_lls, post_lls, pre_probs,
                         post_probs):
                all_scores.append(p)
                if p[0] in label:
                    label_scores.append(p)
            result = label_scores
            pred_dist = [all_scores, label]
        else:
            print('-' * 60)
            print('Probe Sentence {}: {}'.format(j,
                                                 ex['probe_sentences'][
                                                     f'template_{j}'][
                                                     'probe_sentence']))
            print('WARNING: Label not found! {}'.format(label))
            print('         Labels {}'.format(labels))
            for p in zip(labels, pre_lls, post_lls, pre_probs,
                         post_probs):
                print(p)

        # if train_params['COMPUTE_SPECIFICITY']:
        #     assert len(results_specificity) == len(data) - 1, \
        #         (len(results_specificity), len(data))

        output['results'] = result
        output['probs'] = pred_dist
        # output['sim_scores'] = {
        #     'bleu_score': bleu_score,
        #     'bert_score': bert_score,
        #     'bleurt_score': bleurt_score,
        #     'meteor_score': meteor_score,
        # }
        # output['specificity'] = results_specificity
        all_outputs.append(output)
        # bar()

    return all_outputs


def convert_new_token_example(ex):
    word = ex['ent_str']
    nonce = "<{}_new>".format(word.lower())
    modified_definition = re.sub(r"\b({})\b".format(word), nonce, ex['definition'], flags=re.I)
    modified_ent_str = nonce
    j=0
    modified_probe_sentences = {}
    for t in ex['probe_sentences']:
        modified_probe_sentences[t] = {}
        modified_probe_sentences[t]['probe_sentence'] = re.sub(r"\b({})\b".format(word), nonce, ex['probe_sentences'][t]['probe_sentence'], flags=re.I)
        modified_probe_sentences[t]['labels'] = ex['probe_sentences'][t]['labels']
    modified_fields = ['ent_str', 'probe_sentences', 'definition']
    new_ex = {}
    for key in ex:
        if key not in modified_fields:
            new_ex[key] = ex[key]

    new_ex['ent_str'] = modified_ent_str
    new_ex['probe_sentences'] = modified_probe_sentences
    new_ex['definition'] = modified_definition
    return new_ex

def to_tsr_emb_gen(tokenizerTask, tokenizerMLM, ex, device, prepend_def=False,
                                prepend_sent=False, random_def=None)
    '''This function supports a single example only (i.e., bsize=1).'''

    definition = [ex['definition']]
    left_context = [ex['left_context']]
    right_context = [ex['right_context']]
    probe_labels = [v['gpt_labels'] for _, v in ex['probe_sentences'].items()]

    if random_def is not None:
        fake_def = random.choice(random_def)
        probe_sentences = [fake_def + ' ' + v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
        left_context_ps = [fake_def + ' ' + v['left_context_ps'] for _, v
                           in ex['probe_sentences'].items()]
        probe_labels = [[fake_def + ' ' + l for l in pl] for pl in
                        probe_labels]

    elif prepend_def and not prepend_sent:
        probe_sentences = [definition[0] + ' ' + v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
        left_context_ps = [definition[0] + ' ' + v['left_context_ps'] for _, v
                           in ex['probe_sentences'].items()]
        probe_labels = [[definition[0] + ' ' + l for l in pl] for pl in
                        probe_labels]
    elif prepend_sent and not prepend_def:
        probe_sentences = [ex['additional_sent'] + ' ' + v['probe_sentence'] for
                           _, v in ex['probe_sentences'].items()]
        left_context_ps = [ex['additional_sent'] + ' ' + v['left_context_ps']
                           for _, v in ex['probe_sentences'].items()]
        probe_labels = [ex['additional_sent'] + ' ' + pl for pl in probe_labels]
    else:
        probe_sentences = [v['probe_sentence'] for _, v in
                           ex['probe_sentences'].items()]
        left_context_ps = [v['left_context_ps'] for _, v in
                           ex['probe_sentences'].items()]



    right_context_ps = [v['right_context_ps'] for _, v in
                        ex['probe_sentences'].items()]

    definition_tok = tokenizerTask(definition, padding=True, return_tensors="pt")
    def_label_tok = tokenizerTask(definition, padding=True, return_tensors="pt")
    # left_context_tok = tokenizer(left_context, padding=True,
    #                              return_tensors="pt")
    # right_context_tok = tokenizer(right_context, padding=True,
    #                               return_tensors="pt")
    probe_sentences_tok = [
        tokenizerTask(ps, padding=True, return_tensors="pt").to(device) for
        ps in probe_sentences]
    probe_labels_tok = [
        tokenizerTask(pl, padding=True, return_tensors="pt").to(device) for
        pl in probe_labels]
    left_context_ps_tok = [
        tokenizerTask(lc, padding=True, return_tensors="pt").to(device) for
        lc in left_context_ps]
    right_context_ps_tok = [
        tokenizerTask(rc, padding=True, return_tensors="pt").to(device) for
        rc in right_context_ps]

    edit_inner = [{'probe_sentence': ps} for ps in probe_sentences_tok]
    for i, ps in enumerate(edit_inner):
        ps['labels'] = probe_labels_tok[i]
        ps['left_context_ps'] = left_context_ps_tok[i]
        ps['right_context_ps'] = right_context_ps_tok[i]

    def_ = {**definition_tok}
    def_["labels"] = def_label_tok["input_ids"]

    ctx_toks = tokenizerMLM(definition, truncation=True, return_tensors='pt')


    batch = {
        "edit_inner": edit_inner,  # Edit examples
        "definition": def_,  # Locality
        "cond": None,
        "labels": None,
        'contexts': [ctx_toks]
        # "bleu_score": _bleu_score,
        # "bert_score": _bert_score,
        # "bleurt_score": _bleurt_score,
        # "meteor_score": _meteor_score
    }

    return emb_gen_dict_to(batch, device)


def emb_gen_model_no_prepend(batch_pre, batch_post, model, dataset_name=None)

    with torch.no_grad():
        if dataset_name == 'ecbd':
            ex = {}
            ex['input_ids'] = batch_pre["edit_inner"][0]['probe_sentence'][
                'input_ids'][0].unsqueeze(0)
            ex['labels'] = batch_pre["edit_inner"][0]['probe_sentence'][
                'input_ids'][0].unsqueeze(0)  # Dummy label
            ex['attention_mask'] = batch_pre["edit_inner"][0]['probe_sentence'][
                'attention_mask'][0].unsqueeze(0)
            ex['contexts'] = batch_pre['contexts']
        else:
            ex = {}
            ex['input_ids'] = batch_pre["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
            ex['attention_mask'] = batch_pre["edit_inner"][0]['labels'][
                'attention_mask'][0].unsqueeze(0)
            ex['labels'] = batch_pre["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
            ex['contexts'] = batch_pre['contexts']

        pre_edit_logits = model(ex).logits

    # Prepend def
    with torch.set_grad_enabled(False):

        if dataset_name == 'ecbd':
            ex = {}
            ex['input_ids'] = batch_post["edit_inner"][0]['probe_sentence'][
                'input_ids'][0].unsqueeze(0)
            ex['labels'] = batch_post["edit_inner"][0]['probe_sentence'][
                'input_ids'][0].unsqueeze(0)  # Dummy label
            ex['attention_mask'] = batch_post["edit_inner"][0][
                'probe_sentence']['attention_mask'][0].unsqueeze(0)
            ex['contexts'] = batch_pre['contexts']

        else:
            ex = {}
            ex['input_ids'] = batch_post["edit_inner"][0]['labels'][
                'input_ids'][
                0].unsqueeze(0)
            ex['attention_mask'] = batch_post["edit_inner"][0]['labels'][
                'attention_mask'][0].unsqueeze(0)
            ex['labels'] = batch_post["edit_inner"][0]['labels']['input_ids'][
                0].unsqueeze(0)
            ex['contexts'] = batch_pre['contexts']

        post_edit_logits = model(ex).logits


    with torch.no_grad():
        n_probe_labels = batch_pre['edit_inner'][0]['labels']['input_ids'].size(
            0)
        pre_edit_dict = []
        post_edit_dict = []

        for i in range(n_probe_labels):
            if dataset_name == 'ecbd':
                pre_label = \
                batch_pre["edit_inner"][0]["probe_sentence"]['input_ids'][
                    i].unsqueeze(0)
                post_label = \
                batch_post["edit_inner"][0]['probe_sentence']['input_ids'][
                    i].unsqueeze(0)
            else:
                pre_label = \
                batch_pre["edit_inner"][0]['labels']['input_ids'][
                    i].unsqueeze(0)
                post_label = \
                batch_post["edit_inner"][0]['labels']['input_ids'][
                    i].unsqueeze(0)

            pre_edit_dict.append(
                get_log_probs(pre_edit_logits, pre_label, shift=True))

            post_edit_dict.append(
                get_log_probs(post_edit_logits, post_label, shift=True))

    post_loc_dict = None
    pre_loc_dict = None

    return pre_edit_logits, post_edit_logits, pre_edit_dict, post_edit_dict, \
           post_loc_dict, pre_loc_dict
