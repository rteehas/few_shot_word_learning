import itertools
from easyeditor import BaseEditor
from easyeditor import IKEHyperParams
from easyeditor.models.ike.util import encode_ike_facts
from sentence_transformers import SentenceTransformer
from ..llama_eval import prepare_type_1_fewshot, prepare_for_type_2_fewshot, get_sentence_probs, evaluate_type_1, evaluate_type_2, filter_gre
import torch
import uuid
from datasets import load_from_disk
import json
from functools import partial
import numpy as np
import re

def add_new_token(editor):
    editor.tok.add_tokens(['<nonce>'])
    editor.model.resize_token_embeddings(len(editor.tok))
    return editor

def ike_edit(ground_truth, target_definition):
    definition_prompt = "The word <nonce> is defined as"
    rephrased_definition_prompt = "The word <nonce> means"

    hparams = IKEHyperParams.from_hparams('./hparams/IKE/llama-7b.yaml')
    editor = BaseEditor.from_hparams(hparams)
    editor = add_new_token(editor)

    sentence_model = SentenceTransformer(hparams.sentence_model_name)

    train_ds = [
        {
            "prompt": definition_prompt,
            "target_new": target_definition,
            "rephrase_prompt": rephrased_definition_prompt,
        }
    ]

    encode_ike_facts(sentence_model, train_ds, hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=[definition_prompt],
        ground_truth=[ground_truth],
        rephrase_prompts=[rephrased_definition_prompt],  # new para
        target_new=[target_definition],
        subject=['<nonce>'],
        train_ds=train_ds,
        copy=True,
        return_orig_weights=True,
        keep_original_weight=True,
    )
    print(metrics)
    return edited_model, editor.tok


def eval_ke_baseline(ex, sents, defs, with_definition=False, with_prompt=False):
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
        seqs = seqs[1:]
        labels = labels[1:]
        base_seqs = base_seqs[1:]
        samples = samples[1:]

    print("target definitions", target_definitions)
    
    total_probs = []
    if with_prompt:
        for sample, seq, base_seq, target_definition in zip(samples, seqs, base_seqs, target_definitions):
            model, tokenizer = ike_edit(ground_truth=ground_truth_definition, target_definition=target_definition)
            with torch.no_grad():
                model.eval()
                prob = get_sentence_probs(model, tokenizer, [seq], [base_seq])
                total_probs.append(prob)
    else:
        for sample, seq, target_definition in zip(samples, seqs, target_definitions):
            model, tokenizer = ike_edit(ground_truth=ground_truth_definition, target_definition=target_definition)
            with torch.no_grad():
                model.eval()
                toks = tokenizer(seq, return_tensors="pt").to(model.device)
                label = toks['input_ids'].clone()
                out = model(input_ids = toks['input_ids'], attention_mask=toks['attention_mask'], labels=label)
                # prob = get_sentence_probs(model, tokenizer, [seq], [base_seq])
                prob = -out.loss.item()
                total_probs.append(prob)

    if ex["ANSWER_TYPE"] == "top_1":
        return evaluate_type_1(total_probs, labels)
    elif ex["ANSWER_TYPE"] == "top_2":
        return evaluate_type_2(total_probs, labels)

def run_ke_baseline():
    args = get_arguments().parse_args()
    path = args.path
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
            for ex in subselection['train']:
                # try:
                curr_sent_dict = {}
                base_sent_dict = selected_sent_dict[ex["QUESTION"]]
                for key in base_sent_dict:
                    if with_def:
                        curr_sent_dict[key] = base_sent_dict[key][:k]
                    else:
                        curr_sent_dict[key] = base_sent_dict[key][:k + 1]
                outputs.append(eval_ke_baseline(ex=ex, 
                                                sents=curr_sent_dict, 
                                                defs=defs, 
                                                with_definition=with_def, 
                                                with_prompt=with_prompt))

                acc = sum(outputs) / len(outputs)
                print("Accuracy for k = {} is {}".format(k, acc))
                if k in scores:
                    scores[k].append(acc)
                else:
                    scores[k] = [acc]

    print("Across Trials Results")
    for value in scores:
        print("Accuracy for {}".format(value))
        print("{} ({})".format(round(np.mean(np.array(scores[value])), 4), np.std(np.array(scores[value]))))

    fname = "ike_with_prompt_{}_with_def_{}.json".format(args.with_prompt, with_def, args.tuning)

    with open(fname, 'w') as fp:
        json.dump(scores, fp)

    return scores
