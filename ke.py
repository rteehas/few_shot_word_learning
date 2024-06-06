from argparse import ArgumentParser
import itertools
from EasyEdit.easyeditor import BaseEditor
from EasyEdit.easyeditor import IKEHyperParams, ROMEHyperParams, MENDHyperParams
from EasyEdit.easyeditor.models.ike.util import encode_ike_facts
from EasyEdit.easyeditor import ZsreDataset
from sentence_transformers import SentenceTransformer
import torch
import uuid
from datasets import load_from_disk
import json
from functools import partial
import numpy as np
import re
import sys
import os
from tqdm import tqdm

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
from llama_eval import prepare_type_1_fewshot, prepare_for_type_2_fewshot, get_sentence_probs, evaluate_type_1, evaluate_type_2, filter_gre


def add_new_token(editor):
    editor.tok.add_tokens(['<nonce>'])
    editor.model.resize_token_embeddings(len(editor.tok))
    return editor

def ike_edit(editor, ground_truth, target_definition):
    definition_prompt = "The word <nonce> is defined as"
    rephrased_definition_prompt = "The word <nonce> means"

    # hparams = IKEHyperParams.from_hparams('EasyEdit/hparams/IKE/llama-7b.yaml')
    # editor = BaseEditor.from_hparams(hparams)
    # editor = add_new_token(editor)

    # sentence_model = SentenceTransformer(hparams.sentence_model_name)

    # train_ds = [
    #     {
    #         "prompt": definition_prompt,
    #         "target_new": target_definition,
    #         "rephrase_prompt": rephrased_definition_prompt,
    #     }
    # ]
    train_ds = ZsreDataset('EasyEdit/data/zsre/zsre_mend_train.json')

    # encode_ike_facts(sentence_model, train_ds, hparams)
    metrics, edited_model, _, icl = editor.edit(
        prompts=[definition_prompt],
        ground_truth=[ground_truth],
        rephrase_prompts=[rephrased_definition_prompt],  # new para
        target_new=[target_definition],
        subject=['<nonce>'],
        train_ds=train_ds,
        copy=True,
        return_orig_weights=True,
        keep_original_weight=True,
        verbose=False
    )
    # print(metrics)
    return edited_model, editor.tok, icl

def rome_edit(editor, target_definition):
    prompts = ["The word <nonce> is defined as"]
    target_new = [target_definition]
    subject = ['<nonce>']
    # hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/llama-7b.yaml')
    # editor = BaseEditor.from_hparams(hparams)
    # editor = add_new_token(editor)

    metrics, edited_model, weights_copy = editor.edit(
        prompts=prompts,
        ground_truth=None,
        target_new=target_new,
        subject=subject,
        keep_original_weight=True,
        return_orig_weights=True
    )
    # print(metrics)
    return edited_model, editor.tok, weights_copy

def mend_edit(editor, ground_truth, target_definition):
    prompts = ["The word <nonce> is defined as"]
    target_new = [target_definition]
    # hparams = MENDHyperParams.from_hparams('EasyEdit/hparams/MEND/llama-7b.yaml')
    # editor = BaseEditor.from_hparams(hparams)
    # editor = add_new_token(editor)

    metrics, edited_model, weights_copy = editor.edit(
        prompts=prompts,
        ground_truth=[ground_truth],
        target_new=target_new,
        sequential_edit=False,
        return_orig_weights=True
    )
    # print(metrics)
    return edited_model, editor.tok, weights_copy

def get_hparams_and_editor(method):
    if method == "IKE":
        hparams = IKEHyperParams.from_hparams('EasyEdit/hparams/IKE/llama-7b.yaml')
        editor = BaseEditor.from_hparams(hparams)
        editor = add_new_token(editor)
    elif method == "MEND":
        hparams = MENDHyperParams.from_hparams('EasyEdit/hparams/MEND/llama-7b.yaml')
        editor = BaseEditor.from_hparams(hparams)
        editor = add_new_token(editor)
    elif method == "ROME":
        hparams = ROMEHyperParams.from_hparams('EasyEdit/hparams/ROME/llama-7b.yaml')
        editor = BaseEditor.from_hparams(hparams)
        editor = add_new_token(editor)
    else:
        raise NotImplementedError("the method {} is not implemented".format(method))
    return hparams, editor


def eval_ke_baseline(ex, sents, defs, editor, method, with_definition=False, with_prompt=False):
    ground_truth_definition = "a Japanese company that is known for its innovative products and services."
    if with_prompt:
        if ex["ANSWER_TYPE"] == "top_1":
            seqs, labels, base_seqs, samples = prepare_type_1_fewshot(ex, sents, with_definition, defs, with_prompt=True)
        elif ex["ANSWER_TYPE"] == "top_2":
            seqs, labels, base_seqs, samples = prepare_for_type_2_fewshot(ex, sents, with_definition, defs, with_prompt=True)
        else:
            raise NotImplementedError
    else:
        if ex["ANSWER_TYPE"] == "top_1":
            seqs, labels, samples = prepare_type_1_fewshot(ex, sents, with_definition, defs, with_prompt=False)
        elif ex["ANSWER_TYPE"] == "top_2":
            seqs, labels, samples = prepare_for_type_2_fewshot(ex, sents, with_definition, defs, with_prompt=False)
        else:
            raise NotImplementedError
        
    target_definitions = [s[0] for s in samples]
    if not with_definition:
        samples = [s[1:] for s in samples]
        print(samples)
        print("k = ", [len(s) for s in samples])



    print("target definitions", target_definitions)
    
    total_probs = []
    if with_prompt:
        for sample, seq, base_seq, target_definition in zip(samples, seqs, base_seqs, target_definitions):
            if method == "IKE":
                model, tokenizer, icl = ike_edit(editor=editor,
                                                ground_truth=ground_truth_definition, 
                                                target_definition=target_definition)
                ike_icl_examples = icl[0]
                new_seq = ''.join(ike_icl_examples) + " {}".format(seq)
            elif method == "ROME":
                new_seq = seq
                model, tokenizer, weights_copy = rome_edit(editor=editor,
                                                target_definition=target_definition)
            elif method == "MEND":
                new_seq = seq
                model, tokenizer, weights_copy = mend_edit(editor=editor,
                                                           ground_truth=ground_truth_definition,
                                                           target_definition=target_definition)

            # print(new_seq)
            with torch.no_grad():
                if method == "MEND":
                    model = model.model
                model.eval()
                prob = get_sentence_probs(model, tokenizer, [new_seq], [base_seq])
                total_probs.append(prob)
            
            if method in ["ROME", "MEND"]:
                assert weights_copy != {}, "weights copy must not be empty"
                print("modified weights = ", list(weights_copy.keys()))
                editor.model.load_state_dict(weights_copy, strict=False)
    else:
        for sample, seq, target_definition in zip(samples, seqs, target_definitions):
            if method == "IKE":
                model, tokenizer, icl = ike_edit(editor=editor,
                                                ground_truth=ground_truth_definition, 
                                                target_definition=target_definition)
                ike_icl_examples = icl[0]
                new_seq = ''.join(ike_icl_examples) + " {}".format(seq)

            elif method == "ROME":
                new_seq = seq
                model, tokenizer, weights_copy = rome_edit(editor=editor,
                                                target_definition=target_definition)
            elif method == "MEND":
                new_seq = seq
                model, tokenizer, weights_copy = mend_edit(editor=editor,
                                                           ground_truth=ground_truth_definition,
                                                           target_definition=target_definition)
            # print(new_seq)
            with torch.no_grad():
                model.eval()
                if method == "IKE":
                    prob = get_sentence_probs(model, tokenizer, [new_seq], [seq])
                else:
                    if method == "MEND":
                        toks = tokenizer(seq, return_tensors="pt").to(model.config.device)
                    else:
                        toks = tokenizer(seq, return_tensors="pt").to(model.device)
                    label = toks['input_ids'].clone()
                    if method == "MEND":
                        out = model.model(input_ids = toks['input_ids'], attention_mask=toks['attention_mask'], labels=label)
                    else:
                        out = model(input_ids = toks['input_ids'], attention_mask=toks['attention_mask'], labels=label)
                    # prob = get_sentence_probs(model, tokenizer, [seq], [base_seq])
                    prob = -out.loss.item()
                total_probs.append(prob)
            
            if method in ["ROME", "MEND"]:
                assert weights_copy != {}, "weights copy must not be empty"
                print("modified weights = ", list(weights_copy.keys()))
                editor.model.load_state_dict(weights_copy, strict=False)


    if ex["ANSWER_TYPE"] == "top_1":
        return evaluate_type_1(total_probs, labels)
    elif ex["ANSWER_TYPE"] == "top_2":
        print(total_probs)
        if type(total_probs[0]) == list:
            return evaluate_type_2([t[0] for t in total_probs], labels)
        else:
            return evaluate_type_2(total_probs, labels)

def run_ke_baseline():
    args = get_arguments().parse_args()
    print(args)
    gre = load_from_disk("processed_kaplan_v0")
    id = uuid.uuid4()
    subselection = gre.filter(lambda ex: "(i)" not in ex['QUESTION'])
    if args.defs != '':
        with open(args.defs, 'r') as fp:
            defs = json.load(fp)
            # with_def = True
            subselection = subselection.filter(partial(filter_gre, defs))
            with_def = args.with_def
    else:
        defs = None
        with_def = False
    
    with_prompt = args.with_prompt
    answers = subselection['train']['ANSWERS']
    answers = list(itertools.chain(*answers))
    answers = list(itertools.chain(*answers))

    with open(args.sents, 'r') as fp:
        sents = json.load(fp)

    if args.sent_version == "answer":
        with open("gre_examples_gpt4.json", 'r') as fp:
            auxiliary_sents = json.load(fp)

    scores = {}
    max_k = 6
    selected_sent_dict = {}
            # sent_dict = sents
            # for key in sent_dict:
            #     if key in auxiliary_sents[ex['QUESTION']] and len(sent_dict[key]) < 10:
            #         sent_dict[key] += auxiliary_sents[ex['QUESTION']][key]
    method = args.ke_method
    # hparams, editor = get_hparams_and_editor(method = method)

    for trial in range(args.trials):
        for ex in subselection['train']:
            if args.sent_version == "question":
                sent_dict = sents[ex['QUESTION']]
                for key in sent_dict:
                    if defs is not None:
                        samples = np.random.choice(
                            [s for s in sent_dict[key] if
                             re.search(r"\b({})\b".format(key), s, flags=re.I) is not None], size=max_k,
                            replace=False).tolist()

                        if key in defs:
                            definition = defs[key]
                        else:
                            definition = defs[key.lower()]

                        def_s = "The word {} is defined as {}".format("<nonce>", definition)
                        samples = [def_s] + samples
                        sent_dict[key] = samples
                    else:
                        raise NotImplementedError
                    
                selected_sent_dict[ex["QUESTION"]] = sent_dict

            elif args.sent_version == "answer":
                raise NotImplementedError
        
        for k in range(1, max_k):
            print("k = {}".format(k))
            outputs = []
            for ex in tqdm(subselection['train'], total=len(subselection['train'])):
                # try:
                curr_sent_dict = {}
                base_sent_dict = selected_sent_dict[ex["QUESTION"]]
                print("base", base_sent_dict)
                for key in base_sent_dict:
                    if with_def:
                        curr_sent_dict[key] = base_sent_dict[key][:k]
                    else:
                        curr_sent_dict[key] = base_sent_dict[key][:k + 1]
                print("current", curr_sent_dict)
    #             outputs.append(eval_ke_baseline(ex=ex, 
    #                                             sents=curr_sent_dict, 
    #                                             defs=defs,
    #                                             editor=editor,
    #                                             method=method,
    #                                             with_definition=with_def, 
    #                                             with_prompt=with_prompt))
                
    #             acc_so_far = sum(outputs) / len(outputs)
    #             print("Accuracy So Far for k = {} is {}".format(k, acc_so_far))


    #         acc = sum(outputs) / len(outputs)
    #         print("Accuracy for k = {} is {}".format(k, acc))
    #         if k in scores:
    #             scores[k].append(acc)
    #         else:
    #             scores[k] = [acc]

    # print("Across Trials Results")
    # for value in scores:
    #     print("Accuracy for {}".format(value))
    #     print("{} ({})".format(round(np.mean(np.array(scores[value])), 4), np.std(np.array(scores[value]))))

    # fname = "{}_with_prompt_{}_with_def_{}.json".format(args.ke_method, args.with_prompt, with_def)

    # with open(fname, 'w') as fp:
    #     json.dump(scores, fp)

    # return scores

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--sents", type=str, default='gre_examples_gpt4_v2.json')
    parser.add_argument("--defs", type=str, default='gre_definitions_all.json')
    parser.add_argument("--sent_version", type=str)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--with_prompt", action="store_true")
    parser.add_argument("--with_def", action="store_true")
    parser.add_argument("--ke_method", type=str)
    return parser

if __name__ == "__main__":
    run_ke_baseline()