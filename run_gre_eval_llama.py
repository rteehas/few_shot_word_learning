from llama_eval import *
from train_with_llama import *
from transformers import RobertaForMaskedLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from datasets import load_from_disk
from functools import partial
import json
from argparse import ArgumentParser
import numpy as np

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--sents", type=str, required=True)
    parser.add_argument("--defs", type=str, default='')
    parser.add_argument("--sent_version", type=str)
    return parser

def main():
    args = get_arguments().parse_args()
    path = args.path
    gre = load_from_disk("processed_kaplan_v0")
    subselection = gre.filter(lambda ex: "(i)" not in ex['QUESTION'])


    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.sents, 'r') as fp:
        sents = json.load(fp)

    if "pile" in args.sents:
        with open("gre_examples_gpt4.json", 'r') as fp:
            auxiliary_sents = json.load(fp)

    tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
    tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
    nonces = list(tokenizerTask.get_added_vocab().keys())
    tokenizerMLM.add_tokens(nonces)
    tokenizerTask.add_tokens(nonces)
    firstLM = RobertaForMaskedLM.from_pretrained("roberta-base", low_cpu_mem_usage=True)
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", low_cpu_mem_usage=True)
    firstLM.resize_token_embeddings(len(tokenizerMLM))
    secondLM.resize_token_embeddings(len(tokenizerTask))
    if "mean" in path:
        memory_config = AggregatorConfig()
    elif "cls" in path:
        memory_config = TransformerCLSConfig(
            input_size=firstLM.config.hidden_size,
            nhead=firstLM.config.num_attention_heads,
            num_layers=1
        )

    mask_token_id = tokenizerMLM.mask_token_id
    model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), [-1], mask_token_id, memory_config, 1, None).to(device)
    model.load_state_dict(torch.load(path + "/pytorch_model.bin"))
    model.device = device
    with torch.no_grad():
        scores = {}
        for trial in range(3):
            for k in range(1, 7):
                outputs = []
                for ex in subselection['train']:
                    if args.sent_version == "question":
                        sent_dict = sents[ex['QUESTION']]
                    elif args.sent_version == "answer":
                        sent_dict = sents
                        for k in sent_dict:
                            if k in auxiliary_sents[ex['QUESTION']] and len(sent_dict[k]) < 10:
                                sent_dict[k] += auxiliary_sents[ex['QUESTION']][k]
                    outputs.append(evaluate_emb_gen(model, tokenizerMLM, tokenizerTask, ex, sent_dict,k))
                acc = sum(outputs) / len(outputs)
                print("Accuracy for k = {} is {}".format(k, acc))
                if k in scores:
                    scores[k].append(acc)
                else:
                    scores[k] = [acc]


    for value in scores:
        print("{} ({})".format(round(np.mean(np.array(scores[value])), 4), round(np.std(np.array(scores[value]))), 4))
