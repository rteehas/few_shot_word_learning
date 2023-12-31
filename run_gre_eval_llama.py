from llama_eval import *
from train_with_llama import *
from transformers import RobertaForMaskedLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from datasets import load_from_disk
from functools import partial
import json
from argparse import ArgumentParser
import numpy as np
import itertools
import re

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--sents", type=str, required=True)
    parser.add_argument("--defs", type=str, default='')
    parser.add_argument("--sent_version", type=str)
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

    path = "model_checkpoints/layers/no_mp/llama/input_and_output/filtered/pile/layernorm/{}_layers/last_{}/{}_batch_size/{}_agg/{}_examples/lr_{}/weight_decay_{}/{}/"
    path = path.format(args.num_layers, args.num_feature_layers, args.batch_size * args.gradient_accumulation_steps, args.memory,
                       args.num_examples, args.lr, args.weight_decay, neg_string)

    if args.negative_examples and args.regression_objective:
        alpha_str = "distillation_weight_{}_temp_{}/".format(args.regression_alpha, args.distillation_temp)
        hidden_str = "output_embedding_cosine/"

        path = path + alpha_str + hidden_str

    suffix = "checkpoints/"
    if os.path.isdir(path + suffix):
        suffix = "checkpoints2/"
    path = path + suffix
    os.makedirs(path, exist_ok=True)

    return path


def extract_arguments_from_path(path):
    args = {}

    # Adjusting the regex pattern to include num_feature_layers
    if "last" in path:
        regex = r"model_checkpoints/layers/no_mp/llama/input_and_output/filtered/interleaved/layernorm/(\d+)_layers/last_(\d+)/(\d+)_batch_size/(\w+)_agg/(\d+)_examples/lr_([0-9.]+)/weight_decay_([0-9.]+)/(\w+)"
        match = re.search(regex, path)

        if match:
            args['num_layers'] = int(match.group(1))
            args['num_feature_layers'] = int(match.group(2))
            args['batch_size'] = int(match.group(2))  # Adjust this if you need to divide by gradient_accumulation_steps
            args['memory'] = match.group(4)
            args['num_examples'] = int(match.group(5))
            args['lr'] = float(match.group(6))
            args['weight_decay'] = float(match.group(7))

            # neg_string = match.group(8)
            # if neg_string == "with_negatives":
            #     args['negative_examples'] = True
            #     args['regression_objective'] = False
            # else:
            #     # Assuming other cases are not relevant as they are not represented in the example path
            #     args['negative_examples'] = False
            #     args['regression_objective'] = False
    else:
        regex = r"model_checkpoints/layers/no_mp/llama/input_and_output/filtered/pile/layernorm/(\d+)_layers/(\d+)_batch_size/(\w+)_agg/(\d+)_examples/lr_([0-9.]+)/weight_decay_([0-9.]+)/(\w+)"
        match = re.search(regex, path)

        if match:
            args['num_layers'] = int(match.group(1))
            # args['num_feature_layers'] = int(match.group(2))
            args['batch_size'] = int(match.group(2))  # Adjust this if you need to divide by gradient_accumulation_steps
            args['memory'] = match.group(3)
            args['num_examples'] = int(match.group(4))
            args['lr'] = float(match.group(5))
            args['weight_decay'] = float(match.group(6))

            # neg_string = match.group(8)
            # if neg_string == "with_negatives":
            #     args['negative_examples'] = True
            #     args['regression_objective'] = False
            # else:
            #     # Assuming other cases are not relevant as they are not represented in the example path
            #     args['negative_examples'] = False
            #     args['regression_objective'] = False

    return args

def main():
    args = get_arguments().parse_args()
    path = args.path
    gre = load_from_disk("processed_kaplan_v0")
    subselection = gre.filter(lambda ex: "(i)" not in ex['QUESTION'])

    answers = subselection['train']['ANSWERS']
    answers = list(itertools.chain(*answers))
    answers = list(itertools.chain(*answers))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.sents, 'r') as fp:
        sents = json.load(fp)

    if "pile" in args.sents:
        for key in sents:
            for i,example in enumerate(sents[key]):
                split = example.split(".")
                output = [idx for idx, element in enumerate(split) if
                          re.search(r"\b({})\b".format(key), element, flags=re.I) is not None]
                first_index = output[0]

                new_text = ".".join(split[first_index:])
                sents[key][i] = new_text

    if args.sent_version == "answer":
        with open("gre_examples_gpt4.json", 'r') as fp:
            auxiliary_sents = json.load(fp)

    tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
    tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
    nonces = list(tokenizerTask.get_added_vocab().keys())
    # tokenizerMLM.add_tokens(nonces)
    # tokenizerTask.add_tokens(nonces)
    firstLM = RobertaForMaskedLM.from_pretrained("roberta-large", low_cpu_mem_usage=True)
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", low_cpu_mem_usage=True)
    #firstLM.resize_token_embeddings(len(tokenizerMLM))
    #secondLM.resize_token_embeddings(len(tokenizerTask))

    # config_args = extract_arguments_from_path(args.path)
    # print(config_args)
    # if config_args['memory'] == "mean":
    #     memory_config = AggregatorConfig()
    # elif config_args['memory'] == 'cls':
    memory_config = TransformerCLSConfig(
        input_size=firstLM.config.hidden_size,
        nhead=2,
        num_layers=1
    )

    mask_token_id = tokenizerMLM.mask_token_id
    # if 'num_feature_layers' in config_args:
    #     layers = [-1 * (x + 1) for x in range(config_args['num_feature_layers'])]
    # else:
    #     layers=[-1]
    layers = [-1]
    model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), layers, mask_token_id, memory_config, 1, None).to(device)
    model.emb_gen.load_state_dict(torch.load(path + "/pytorch_model.bin"))
    model.device = device
    model.firstLM.eval()
    model.secondLM.eval()

    # new_nonces = list(map(lambda w: "<{}_new>".format(w.lower()), answers))
    # new_nonces = list(set(new_nonces))
    # tokenizerMLM.add_tokens(new_nonces)
    # tokenizerTask.add_tokens(new_nonces)
    # new_token_num = len(list(tokenizerTask.get_added_vocab().keys())) - len(nonces)

    # model.firstLM.resize_token_embeddings(len(tokenizerMLM))
    # model.secondLM.resize_token_embeddings(len(tokenizerTask))
    # model.add_new_tokens(new_token_num)
    model.eval()
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
                        for key in sent_dict:
                            if key in auxiliary_sents[ex['QUESTION']] and len(sent_dict[key]) < 10:
                                sent_dict[key] += auxiliary_sents[ex['QUESTION']][key]
                    outputs.append(evaluate_emb_gen(model, tokenizerMLM, tokenizerTask, ex, sent_dict,k))
                    #except:
                     #   continue
                acc = sum(outputs) / len(outputs)
                print("Accuracy for k = {} is {}".format(k, acc))
                if k in scores:
                    scores[k].append(acc)
                else:
                    scores[k] = [acc]


    for value in scores:
        print("{} ({})".format(round(np.mean(np.array(scores[value])), 4), np.std(np.array(scores[value]))))

if __name__ == "__main__":
    main()
