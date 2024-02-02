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
from init_baseline import *
from w2v_baselines import *
import uuid

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--sents", type=str, required=True)
    parser.add_argument("--defs", type=str, default='')
    parser.add_argument("--sent_version", type=str)
    parser.add_argument("--init_method", type=str, default="random")
    parser.add_argument("--setting", type=str, default="emb_gen")
    parser.add_argument("--tuning", action="store_true")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--with_prompt", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
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
        regex = r"model_checkpoints/layers/no_mp/llama/input_and_output/filtered/pile/layernorm/(\d+)_layers/last_(\d+)/(\d+)_batch_size/(\w+)_agg/(\d+)_examples/lr_([0-9.]+)/weight_decay_([0-9.]+)/(\w+)"
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

def gradient_descent_tuning_gre(model, tokenizer, ex, lr, max_num_steps):
    new_tok_index = tokenizer.convert_tokens_to_ids("<nonce>")
    zero_grad_indices = torch.arange(0, len(tokenizer)) != new_tok_index
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True
    for param in model.get_output_embeddings().parameters():
        param.requires_grad = True

    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=lr)

    pass

def eval_baseline(args):
    args = get_arguments().parse_args()
    path = args.path
    gre = load_from_disk("processed_kaplan_v0")
    id = uuid.uuid4()
    subselection = gre.filter(lambda ex: "(i)" not in ex['QUESTION'])
    if args.defs != '':
        with open(args.defs, 'r') as fp:
            defs = json.load(fp)
            with_def = True
            subselection = subselection.filter(partial(filter_gre, defs))
    else:
        defs = None
        with_def = False

    answers = subselection['train']['ANSWERS']
    answers = list(itertools.chain(*answers))
    answers = list(itertools.chain(*answers))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.sents, 'r') as fp:
        sents = json.load(fp)

    if "pile" in args.sents:
        for key in sents:
            for i, example in enumerate(sents[key]):
                split = example.split(".")
                output = [idx for idx, element in enumerate(split) if
                          re.search(r"\b({})\b".format(key), element, flags=re.I) is not None]
                first_index = output[0]

                new_text = ".".join(split[first_index:])
                sents[key][i] = new_text

    if args.sent_version == "answer":
        with open("gre_examples_gpt4.json", 'r') as fp:
            auxiliary_sents = json.load(fp)


    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                low_cpu_mem_usage=True).to(device)
    tokenizerTask = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", legacy=True,
                                                   use_fast=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizerTask.add_tokens(["<nonce>"])
    # nonces = list(map(lambda w: "<{}_new>".format(w), answers))
    # nonces = list(set(nonces))
    # secondLM.resize_token_embeddings(len(tokenizerTask))
    nonces = ["<nonce>"]
    if args.init_method == "random":
        secondLM, tokenizerTask = default_init(secondLM, tokenizerTask, nonces)
    elif args.init_method == "mean":
        secondLM, tokenizerTask = mean_init(secondLM, tokenizerTask, nonces)
    elif args.init_method == "zero":
        secondLM, tokenizerTask = zero_init(secondLM, tokenizerTask, nonces)
    elif args.init_method == "random_mean":
        secondLM, tokenizerTask = random_mean_init(secondLM, tokenizerTask, nonces)

    secondLM.eval()
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
                    if with_def and defs is not None:
                        samples = np.random.choice(
                            [s for s in sent_dict[key] if
                             re.search(r"\b({})\b".format(key), s, flags=re.I) is not None], size=max_k - 1,
                            replace=False).tolist()

                        if key in defs:
                            definition = defs[key]
                        else:
                            definition = defs[key.lower()]

                        def_s = "The word {} is defined as {}".format("<nonce>", definition)
                        samples = [def_s] + samples
                        sent_dict[key] = samples
                    else:
                        samples = np.random.choice(
                            [s for s in sent_dict[key] if
                             re.search(r"\b({})\b".format(key), s, flags=re.I) is not None], size=max_k,
                            replace=False).tolist()
                        sent_dict[key] = samples
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
                    curr_sent_dict[key] = base_sent_dict[key][:k]
                outputs.append(evaluate_baseline_example_fewshot(secondLM, tokenizerTask, ex, curr_sent_dict, with_def, defs, args.tuning, with_prompt=args.with_prompt, lr=args.lr))
                # except:
                #
                #     continue
            if not args.tuning:
                acc = sum(outputs) / len(outputs)
                print("Accuracy for k = {} is {}".format(k, acc))
                if k in scores:
                    scores[k].append(acc)
                else:
                    scores[k] = [acc]
            else:
                step_1_results = [o[0] for o in outputs]
                step_2_results = [o[1] for o in outputs]
                step_1_acc = sum(step_1_results) / len(step_1_results)
                step_2_acc = sum(step_2_results) / len(step_2_results)
                print("Accuracy for step 1 of GD for k= {} is {}".format(k, step_1_acc))
                print("Accuracy for step 2 of GD for k= {} is {}".format(k, step_2_acc))
                key = "k = {}, step = {}"
                step_accs = [step_1_acc, step_2_acc]
                for idx in range(2):
                    k = key.format(k, idx + 1)
                    if k in scores:
                        scores[k].append(step_accs[idx])
                    else:
                        scores[k] = [step_accs[idx]]



    print("Across Trials Results")
    for value in scores:
        print("Accuracy for {}".format(value))
        print("{} ({})".format(round(np.mean(np.array(scores[value])), 4), np.std(np.array(scores[value]))))

    print("Per Trial Results")
    for trial in range(args.trials):
        if not args.tuning:
            trial_vals = [scores[value][trial] for value in scores]
            print("{} ({})".format(round(np.mean(np.array(trial_vals)), 4), np.std(np.array(trial_vals))))


        else:
            for step in range(2):
                print("Accuracy for step {} of GD".format(step + 1))
                trial_vals = [scores[value][trial] for value in scores if "step = {}".format(step + 1) in value]
                print("{} ({})".format(round(np.mean(np.array(trial_vals)), 4), np.std(np.array(trial_vals))))
    if args.tuning:
        fname = "baseline_with_prompt_{}_with_def_{}_tuning_{}_lr_{}".format(args.with_prompt, with_def, args.tuning, args.lr)
    else:
        fname = "baseline_with_prompt_{}_with_def_{}_tuning_{}".format(args.with_prompt, with_def, args.tuning)

    fname = "{}_{}.json".format(fname, id)
    with open(fname, 'w') as fp:
        json.dump(scores, fp)

    return scores

def eval_hice(args):
    device = "cuda"
    hice_path = "HiCE/save/model.pt"
    input_linear_path = "baseline_mappings/input_linear.pt"
    output_linear_path = "baseline_mappings/output_linear.pt"
    w2v_dir = 'HiCE/data/base_w2v/wiki_all.sent.split.model'
    corpus_dir = "HiCE/data/wikitext-103/"

    gre = load_from_disk("processed_kaplan_v0")
    subselection = gre.filter(lambda ex: "(i)" not in ex['QUESTION'])
    if args.defs != '':
        with open(args.defs, 'r') as fp:
            defs = json.load(fp)
            with_def = True
            subselection = subselection.filter(partial(filter_gre, defs))
    else:
        defs = None
        with_def = False

    answers = subselection['train']['ANSWERS']
    answers = list(itertools.chain(*answers))
    answers = list(itertools.chain(*answers))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.sents, 'r') as fp:
        sents = json.load(fp)

    if "pile" in args.sents:
        for key in sents:
            for i, example in enumerate(sents[key]):
                split = example.split(".")
                output = [idx for idx, element in enumerate(split) if
                          re.search(r"\b({})\b".format(key), element, flags=re.I) is not None]
                first_index = output[0]

                new_text = ".".join(split[first_index:])
                sents[key][i] = new_text

    if args.sent_version == "answer":
        with open("gre_examples_gpt4.json", 'r') as fp:
            auxiliary_sents = json.load(fp)


    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                low_cpu_mem_usage=True)
    tokenizerTask = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", legacy=True,
                                                   use_fast=False)

    tokenizerTask.add_tokens(["<nonce>"])

    hice = HiCEBaseline(hice_path, input_linear_path, output_linear_path, secondLM).to(device)
    hice.device = device
    dictionary = load_dictionary(w2v_dir, corpus_dir, 24)
    max_k = 6
    result_dict = {}
    for trial in range(10):
        selected_sent_dict = {}
        for ex in subselection['train']:
            if args.sent_version == "question":
                sent_dict = sents[ex['QUESTION']]
                for key in sent_dict:
                    if with_def and defs is not None:
                        samples = np.random.choice(
                            [s for s in sent_dict[key] if
                             re.search(r"\b({})\b".format(key), s, flags=re.I) is not None], size=max_k - 1,
                            replace=False)

                        if key in defs:
                            definition = defs[key]
                        else:
                            definition = defs[key.lower()]

                        def_s = "The word {} is defined as {}".format("<nonce>", definition)
                        samples = [def_s] + samples
                        sent_dict[key] = samples
                    else:
                        samples = np.random.choice(
                            [s for s in sent_dict[key] if
                             re.search(r"\b({})\b".format(key), s, flags=re.I) is not None], size=max_k,
                            replace=False)
                        sent_dict[key] = samples
                selected_sent_dict[ex["QUESTION"]] = sent_dict

        trial_results = []
        for k in range(1, max_k):
            print("k = {}".format(k))
            outputs = []
            for ex in subselection['train']:
                curr_sent_dict = {}
                base_sent_dict = selected_sent_dict[ex["QUESTION"]]
                for key in base_sent_dict:
                    curr_sent_dict[key] = base_sent_dict[key][:k]
                outputs.append(
                    evaluate_hice(hice, tokenizerTask, ex, curr_sent_dict, k, dictionary, with_def=with_def, defs=defs, with_prompt=args.with_prompt))
            acc = sum(outputs) / len(outputs)
            trial_results.append(acc)
            print("Accuracy for k = {} is {}".format(k, acc))
        result_dict[trial] = trial_results
    return result_dict


def main():
    args = get_arguments().parse_args()
    print(args)
    if args.setting == "baseline":
        eval_baseline(args)
    elif args.setting == "emb_gen":
        # id = uuid.uuid4()
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

        if args.defs != '':
            with open(args.defs, 'r') as fp:
                defs = json.load(fp)
                with_def = True
        else:
            defs=None
            with_def = False

        tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
        tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
        nonces = list(tokenizerTask.get_added_vocab().keys())
        # tokenizerMLM.add_tokens(nonces)
        # tokenizerTask.add_tokens(nonces)
        firstLM = RobertaForMaskedLM.from_pretrained("roberta-large", low_cpu_mem_usage=True)
        # T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        # firstLM = T5EncoderModel.from_pretrained("t5-large", low_cpu_mem_usage=True).to(device)
        secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", low_cpu_mem_usage=True)
        # firstLM.resize_token_embeddings(len(tokenizerMLM))
        # secondLM.resize_token_embeddings(len(tokenizerTask))

        # config_args = extract_arguments_from_path(args.path)
        # print(config_args)
        # if config_args['memory'] == "mean":
        memory_config = AggregatorConfig()
        # elif config_args['memory'] == 'cls':
        # memory_config = TransformerCLSConfig(
        #         input_size=firstLM.config.hidden_size,
        #         nhead=2,
        #         num_layers=1
        #     )

        mask_token_id = tokenizerMLM.mask_token_id
        # if 'num_feature_layers' in config_args:
        #     layers = [-1 * (x + 1) for x in range(config_args['num_feature_layers'])]
        # else:
        layers=[-1]
        model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), layers, mask_token_id, memory_config, 1, None, False).to(device)
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
        max_k = 6
        with torch.no_grad():
            scores = {}
            for trial in range(args.trials):
                selected_sent_dict = {}
                for ex in subselection['train']:
                    if args.sent_version == "question":
                        sent_dict = sents[ex['QUESTION']]
                        for key in sent_dict:
                            if with_def and defs is not None:
                                samples = np.random.choice(
                                    [s for s in sent_dict[key] if
                                     re.search(r"\b({})\b".format(key), s, flags=re.I) is not None], size=max_k - 1,
                                    replace=False).tolist()

                                if key in defs:
                                    definition = defs[key]
                                else:
                                    definition = defs[key.lower()]

                                def_s = "The word {} is defined as {}".format("<nonce>", definition)
                                samples = [def_s] + samples
                                sent_dict[key] = samples
                            else:
                                samples = np.random.choice(
                                    [s for s in sent_dict[key] if
                                     re.search(r"\b({})\b".format(key), s, flags=re.I) is not None], size=max_k,
                                    replace=False).tolist()
                                sent_dict[key] = samples
                        selected_sent_dict[ex["QUESTION"]] = sent_dict

                wrong_ans = {}
                for k in range(1,max_k):
                    wrong_ans[k] = []


                for k in range(1, 6):
                    outputs = []
                    for ex in subselection['train']:
                        # try:
                        # if args.sent_version == "question":
                        #     sent_dict = sents[ex['QUESTION']]
                        # elif args.sent_version == "answer":
                        #     sent_dict = sents
                        #     for key in sent_dict:
                        #         if key in auxiliary_sents[ex['QUESTION']] and len(sent_dict[key]) < 10:
                        #             sent_dict[key] += auxiliary_sents[ex['QUESTION']][key]
                        curr_sent_dict = {}
                        base_sent_dict = selected_sent_dict[ex["QUESTION"]]
                        for key in base_sent_dict:
                            curr_sent_dict[key] = base_sent_dict[key][:k]

                        result = evaluate_emb_gen(model, tokenizerMLM, tokenizerTask, ex, curr_sent_dict, k, with_def, defs, with_prompt=args.with_prompt)
                        outputs.append(result)
                        if not result:
                            wrong_ans[k].append(ex["QUESTION"])
                        # except:
                        #     print("ERROR")
                        #     continue
                    acc = sum(outputs) / len(outputs)
                    print("Accuracy for k = {} is {}".format(k, acc))
                    if k in scores:
                        scores[k].append(acc)
                    else:
                        scores[k] = [acc]

        if "negatives" in args.path and "regression" in args.path:
            model_type = "negatives_and_regression"
        elif "negatives" in args.path:
            model_type = "negatives"
        elif "regression" in args.path:
            model_type = "distillation"
        elif "vanilla" in args.path:
            model_type="vanilla"

        with open("embedding_generator_{}_prompt_{}_defs_{}.json".format(model_type, args.with_prompt, with_def), 'w') as fp:
            json.dump(scores, fp)
                # print("-----------Saving Wrong Answers----------")
                # with open("gre_wrong_{}.json".format(trial), 'w') as fp:
                #     json.dump(wrong_ans, fp)

                # print(wrong_ans)



        for value in scores:
            print("{} ({})".format(round(np.mean(np.array(scores[value])), 4), np.std(np.array(scores[value]))))

        print("Per Trial Results")
        for trial in range(args.trials):
            if not args.tuning:
                trial_vals = [scores[value][trial] for value in scores]
                print("{} ({})".format(round(np.mean(np.array(trial_vals)), 4), np.std(np.array(trial_vals))))

        return scores

if __name__ == "__main__":
    main()