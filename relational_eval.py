import re
import json
import string
from collections import OrderedDict
import numpy as np
from train_with_llama import *
from transformers import RobertaForMaskedLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, \
    get_linear_schedule_with_warmup, AdamW, DataCollatorForLanguageModeling, AutoConfig, T5EncoderModel
from tqdm import tqdm
import json
from accelerate import PartialState
import time
import uuid
from copy import deepcopy

def read_jsonl(path: str):
    """Read JSON file. Copied from gsm.py"""
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def reverse_index(arr, value):
    return len(arr) - 1 - arr[::-1].index(value)


def process_explanation(explanation, nonce_mapping=None):
    operations = ["\+", "-", "/", "\*", "="]
    components = [s.strip() for s in re.split("|".join(operations), explanation)]
    masked_examples = []
    if nonce_mapping is None:
        nonce_mapping = OrderedDict()
    final_relation = explanation

    for i, comp in enumerate(components):
        #         print(i, len(nonce_mapping))
        if comp not in nonce_mapping:
            if i == 0 and len(nonce_mapping) == 0:
                nonce = "<nonce>"
            else:
                nonce = "<nonce{}>".format(len(nonce_mapping))
        else:
            nonce = nonce_mapping[comp]
        final_relation = final_relation.replace(comp, nonce, 1)
        masked_examples.append(explanation.replace(comp, nonce, 1))
        if comp not in nonce_mapping:
            nonce_mapping[comp] = nonce
    return masked_examples, final_relation, nonce_mapping


def get_explanations_and_equations(ex):
    keys = list(ex.keys())
    expl_keys = [e for e in keys if "explanation" in e]
    eq_keys = [e for e in keys if "equation" in e]
    explanations = [ex[k] for k in expl_keys]
    equations = [ex[k] for k in eq_keys]
    return explanations, equations


def process_answer(ex, mapping=None, no_text=False):
    explanations, equations = get_explanations_and_equations(ex)
    #     print(explanations, equations)
    if no_text:
        answer = ''
    else:
        answer = ex['answer']
    processed = []
    for i, expl in enumerate(explanations):
        processed_explanation = process_explanation(expl, mapping)

        eq_with_nonces = "{} {}".format(processed_explanation[1], equations[i])
        if no_text:
            answer += eq_with_nonces + "\n"
        else:
            answer = answer.replace(equations[i], eq_with_nonces)
        mapping = processed_explanation[2]
        processed.append((processed_explanation[0], processed_explanation[1]))

    return answer, mapping, processed


def remove_answer_for_eval(answer):
    delimiter = "Final answer:"
    return answer.split(delimiter)[0] + delimiter


def construct_context(mapping):
    context_mapping = OrderedDict()
    for key, value in mapping.items():
        nonce_seq = "{} = {}".format(value, key)

        nonce_context = [nonce_seq]
#         for example in example_list:
#             for sentence in example:
#                 if nonce in sentence:
#                     nonce_context.append(sentence)
        context_mapping[value] = nonce_context
    return list(context_mapping.values())


def create_example(ex, mapping=None, use_one_example=False, no_text=False, let=False, only_let=False, var_names=False,
                   interleaved=False):
    previous_mapping = deepcopy(mapping)
    answer, mapping, processed = process_answer(ex, mapping=mapping, no_text=no_text)
    #     print(answer.split("\n"))
    if let:
        first_let_str = "Let {} = {}"
        second_let_str = first_let_str.lower()

        lets = []
        if var_names:
            alphabet_mapping = deepcopy(mapping)
            alphabet = string.ascii_uppercase
            for i, (k, v) in enumerate(mapping.items()):
                if i <= len(alphabet) - 1:
                    alphabet_mapping[k] = alphabet[i]
                else:
                    new_idx = i % (len(alphabet) - 1)
                    num_iters = i // (len(alphabet) - 1)
                    alphabet_mapping[k] = alphabet[new_idx] + str(num_iters)
        if previous_mapping is not None:
            remaining_mapping = {k: v for k, v in mapping.items() if k not in previous_mapping}
        else:
            remaining_mapping = mapping
        #         print(remaining_mapping)
        if not interleaved:
            for i, (k, v) in enumerate(remaining_mapping.items()):
                if i == 0:
                    if var_names:
                        lets.append(first_let_str.format(k, alphabet_mapping[k]) + ",")
                    else:
                        lets.append(first_let_str.format(k, v) + ",")
                elif i < len(list(remaining_mapping.keys())) - 1:
                    if var_names:
                        lets.append(" " + second_let_str.format(k, alphabet_mapping[k]) + ",")
                    else:
                        lets.append(" " + second_let_str.format(k, v) + ",")
                else:
                    if var_names:
                        lets.append(" " + second_let_str.format(k, alphabet_mapping[k]) + ".\n")
                    else:
                        lets.append(" " + second_let_str.format(k, v) + ".\n")
            if only_let:
                preamble = " ".join(lets) + "Answer: "
            else:
                preamble = " ".join(lets)
            if var_names:
                for k, v in mapping.items():
                    answer = answer.replace(v, alphabet_mapping[k])
            if not only_let:
                answer = preamble + answer
            else:
                answer = preamble
        else:
            split_answer = answer.split("\n")
            split_answer = [s for s in split_answer if s != ""]
            ans_str = ""
            for s in split_answer:
                for i, (k, v) in enumerate(remaining_mapping.items()):
                    if v in s:
                        #                         print("here")
                        if i == 0:
                            if var_names:
                                ans_str += first_let_str.format(k, alphabet_mapping[k]) + ","
                            else:
                                ans_str += first_let_str.format(k, v) + ","
                        elif i < len(list(mapping.keys())) - 1:
                            if var_names:
                                ans_str += " " + second_let_str.format(k, alphabet_mapping[k]) + ","
                            else:
                                ans_str += " " + second_let_str.format(k, v) + ","
                        else:
                            if var_names:
                                ans_str += " " + second_let_str.format(k, alphabet_mapping[k]) + ".\n"
                            else:
                                ans_str += " " + second_let_str.format(k, v) + ".\n"

                ans_str += " " + s + "\n"
                if var_names:
                    for k, v in mapping.items():
                        ans_str = ans_str.replace(v, alphabet_mapping[k])
    if interleaved:
        answer = ans_str

    #     example_list = [p[0] for p in processed]
    if var_names:
        context = construct_context(alphabet_mapping)
    else:
        context = construct_context(mapping)

    if use_one_example:
        context = [[c[0]] for c in context]

    return context, answer, mapping


def process_for_eval(train_set, test_example, k_shot=0, use_one_example=False, no_text=False, let=False, only_let=False, var_names=False, interleaved=False):
    mapping = None
    if k_shot > 0:
        sampled_k_shot_examples = np.random.choice(train_set, size=k_shot, replace=False)
        contexts = []
        answers = []
        for ex in sampled_k_shot_examples:
            context, answer, mapping = create_example(ex, mapping=mapping, use_one_example=use_one_example,
                                                      no_text=no_text, let=let, var_names=var_names)
            answer = "{}\n{}".format(ex['question'].strip("\n"), answer)
            answer = answer + "Final answer: {}".format(ex['final_answer'])
            contexts += context
            answers.append(answer)
    #             print(contexts)

    test_context, test_answer, final_mapping = create_example(test_example, mapping=mapping,
                                                              use_one_example=use_one_example, no_text=no_text, let=let, only_let=only_let, var_names=var_names, interleaved=interleaved)

    test_expl, test_eq = get_explanations_and_equations(test_example)

    # test_answer = "{}\n{}".format(test_example, test_answer)
    # test_answer = test_example['question']
    # truncated_answer = remove_answer_for_eval(test_answer)
    truncated_answer = test_answer.split(test_eq[-1])[0]

    truncated_answer = "{}\n{}".format(test_example['question'], truncated_answer)
    # print(test_answer, test_eq[-1], truncated_answer)
    if k_shot > 0:
        answer_text = "\n".join(answers)
        answer_text = answer_text + "\n" + truncated_answer
        contexts += test_context
        contexts = [c for c in contexts if c != []]  # empty ones are added for the few shot example nonces, remove them
        return contexts, answer_text, test_example['final_answer']
    else:
        return test_context, truncated_answer, test_example['final_answer']


def verify_or_fix_num_tokens(model, tokenizerMLM, tokenizerTask, context):
    tokens_needed_for_task = len(context)
    if len(tokenizerMLM.get_added_vocab()) < tokens_needed_for_task:
        new_tokens = ["<nonce{}>".format(i) for i in range(1, tokens_needed_for_task)]
        tokenizerMLM.add_tokens(new_tokens)
    if len(tokenizerTask.get_added_vocab()) < tokens_needed_for_task:
        new_tokens = ["<nonce{}>".format(i) for i in range(1, tokens_needed_for_task)]
        tokenizerTask.add_tokens(new_tokens)

    if model.num_new_tokens < tokens_needed_for_task:
        model.num_new_tokens = tokens_needed_for_task

def prev_verify_or_fix(model, tokenizerMLM, tokenizerTask, context):
    tokens_needed_for_task = len(context)
    if len(tokenizerMLM.get_added_vocab()) < tokens_needed_for_task:
        new_tokens = ["<nonce{}>".format(i) for i in range(1, tokens_needed_for_task)]
        tokenizerMLM.add_tokens(new_tokens)
    if len(tokenizerTask.get_added_vocab()) < tokens_needed_for_task:
        new_tokens = ["<nonce{}>".format(i) for i in range(1, tokens_needed_for_task)]
        tokenizerMLM.add_tokens(new_tokens)

    if model.num_new_tokens < tokens_needed_for_task:
        model.num_new_tokens = tokens_needed_for_task



def verify_or_fix_baseline_num_tokens(model, tokenizer, context):
    tokens_needed_for_task = len(context)
    if len(tokenizer.get_added_vocab()) < tokens_needed_for_task:
        new_tokens = ["<nonce{}>".format(i) for i in range(1, tokens_needed_for_task)]
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))


def run_example(model, tokenizerMLM, tokenizerTask, train_examples, ex, k_shot, let=False, only_let=False, interleaved=False, mask_new_tokens=False):
    contexts, text, ans = process_for_eval(train_examples, ex, k_shot, use_one_example=False, no_text=True, let=let, only_let=only_let, interleaved=interleaved)
    # print("num new tokens before", len(tokenizerMLM.get_added_vocab()), len(tokenizerTask.get_added_vocab()))
    verify_or_fix_num_tokens(model, tokenizerMLM, tokenizerTask, contexts)
    # print("num new tokens after", len(tokenizerMLM.get_added_vocab()), len(tokenizerTask.get_added_vocab()))

    ctx = [tokenizerMLM(c, padding="longest", truncation=True, return_tensors='pt').to(model.device) for c in contexts]
    target_input = tokenizerTask(text, return_tensors='pt').to(model.device)
    gen_out = generate_multi(model, ctx, target_input['input_ids'], target_input['attention_mask'], 60,
                             mask_new_tokens=mask_new_tokens)
    out_text = tokenizerTask.decode(gen_out[0])
    return out_text, text

def prev_run_example(model, tokenizerMLM, tokenizerTask, train_examples, ex, k_shot, let=False, only_let=False):
    contexts, text, ans = process_for_eval(train_examples, ex, k_shot, use_one_example=False, no_text=True, let=let, only_let=only_let)
    # print("num new tokens before", len(tokenizerMLM.get_added_vocab()), len(tokenizerTask.get_added_vocab()))
    prev_verify_or_fix(model, tokenizerMLM, tokenizerTask, contexts)
    # print("num new tokens after", len(tokenizerMLM.get_added_vocab()), len(tokenizerTask.get_added_vocab()))

    ctx = [tokenizerMLM(c, padding="longest", truncation=True, return_tensors='pt').to(model.device) for c in contexts]
    target_input = tokenizerTask(text, return_tensors='pt').to(model.device)
    gen_out = generate_multi(model, ctx, target_input['input_ids'], target_input['attention_mask'], 60,
                             mask_new_tokens=True, top_k=10)
    out_text = tokenizerTask.decode(gen_out[0])
    return out_text, text


def run_example_baseline(model, tokenizer, train_examples, ex, k_shot, with_relation, let, var_names=False, only_let=False, remove_relation_for_test=False):
    if not with_relation:
        contexts, text, ans = process_for_eval(train_examples, ex, k_shot, use_one_example=False, no_text=True, let=let, only_let= only_let, var_names=var_names)
        verify_or_fix_baseline_num_tokens(model, tokenizer, contexts)
    else:
        text, ans = process_for_baseline_eval(train_examples, ex, k_shot, remove_relation_for_test=remove_relation_for_test)

    target_input = tokenizer(text, return_tensors='pt').to(model.device)
    gen_out = model.generate(**target_input, use_cache=True, top_k=10, max_new_tokens=100)
    out_text = tokenizer.decode(gen_out[0])
    return out_text, text


def run_example_vanilla(model, tokenizer, train_examples, ex, k_shot, remove_relation_for_test=False):
    text, ans = process_for_vanilla_cot(train_examples, ex, k_shot,remove_relation_for_test=remove_relation_for_test)

    target_input = tokenizer(text, return_tensors='pt').to(model.device)
    gen_out = model.generate(**target_input, use_cache=True, top_k=10, max_new_tokens=100)
    out_text = tokenizer.decode(gen_out[0])
    return out_text, text


def process_baseline_answer_with_relation(ex):
    explanations, equations = get_explanations_and_equations(ex)
    answer = ''
    for i, expl in enumerate(explanations):
        eq = equations[i]
        eq_with_relation = "{} {}".format(expl, eq)
#         answer = answer.replace(eq, eq_with_relation)
        answer += eq_with_relation + "\n"
    return answer


def process_for_baseline_eval(train_set, test_example, k_shot=0, remove_relation_for_test=False):
    if k_shot > 0:
        sampled_k_shot_examples = np.random.choice(train_set, size=k_shot, replace=False)
        answers = []
        for ex in sampled_k_shot_examples:
            answer = process_baseline_answer_with_relation(ex)
            answer = "{}\n{}".format(ex['question'], answer)
            answer = answer + "Final answer: {}".format(ex['final_answer'])
            answers.append(answer)
    test_answer = process_baseline_answer_with_relation(test_example)
    test_expl, test_eq = get_explanations_and_equations(test_example)
    truncated_answer = test_answer.split(test_eq[-1])[0]
#     print(truncated_answer)
    if k_shot > 0:
        answer_text = "\n".join(answers)
        if not remove_relation_for_test:
            answer_text = answer_text + "\n" + "{}\n{}".format(test_example['question'], truncated_answer)
        else:
            answer_text = answer_text + "\n" + test_example['question'] + "\n"
        return answer_text, test_example['final_answer']
    else:
        return truncated_answer, test_example['final_answer']


def process_for_vanilla_cot(train_set, test_example, k_shot=0, remove_relation_for_test=False):
    if k_shot > 0:
        sampled_k_shot_examples = np.random.choice(train_set, size=k_shot, replace=False)
        answers = []
        for ex in sampled_k_shot_examples:
            answer = ex['answer']
            answer = "{}\n{}".format(ex['question'], answer)
            answers.append(answer)
    # print(answers)
    truncated_answer = prepare_vanilla_cot(test_example)
    if k_shot > 0:
        answer_text = "\n".join(answers)
        if not remove_relation_for_test:
            answer_text = answer_text + "\n" + "{}\n{}".format(test_example['question'], truncated_answer)
        else:
            answer_text = answer_text + "\n" + test_example['question'] + "\n"
        # answer_text = answer_text + "\n" + truncated_answer
        return answer_text, test_example['final_answer']
    else:
        return truncated_answer, test_example['final_answer']



def prepare_vanilla_cot(ex):
    answer = ex['answer']
    parts = answer.split("\n")
    beginning = parts[:-2]
    end = parts[-2]
    if "<<" in end:
        idx = reverse_index(end, "<<") - 1


        for i, char in enumerate(reversed(end[:idx])):
            if char.isalpha():
                break

        new_idx = idx - i

        return "\n".join(beginning) + "\n" + end[:new_idx] + " "
    else:
        return "\n".join(beginning) + "\n"


def run_vanilla(remove_relation_for_test=False):
    device = "cuda"
    id = uuid.uuid4()
    model = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                             low_cpu_mem_usage=True).to(device)
    with open("annotated_cot_for_relational.json", 'r') as fp:
        train_examples = json.load(fp)

    examples = read_jsonl("test_relation.jsonl")
    model.eval()
    for k_shot in [8]:
        outputs = []
        bad_examples = []
        print("{} shots...".format(k_shot))
        for i, ex in tqdm(enumerate(examples)):
            out_example = {}
            tokenizer = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                       use_fast=False, legacy=True)
            # try:
            out_text, text = run_example_vanilla(model, tokenizer, train_examples, ex, k_shot, remove_relation_for_test=remove_relation_for_test)
            out_example['input'] = text
            out_example['generation'] = out_text
            out_example['k_shot'] = k_shot
            # out_example['final_answer'] = ex['final_answer']
            out_example['original_example'] = ex

            outputs.append(out_example)
            # except:
            #     bad_examples.append(ex)

        with open("relational_vanilla_cot_baseline_{}shot_{}_redo.json".format(k_shot,id), 'w') as fp:
            json.dump(outputs, fp)



def main(path, let=False):
    device = "cuda"
    tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
    tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
    nonces = list(tokenizerTask.get_added_vocab().keys())
    # tokenizerMLM.add_tokens(nonces)
    # tokenizerTask.add_tokens(nonces)
    firstLM = RobertaForMaskedLM.from_pretrained("roberta-large", low_cpu_mem_usage=True)
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                low_cpu_mem_usage=True)
    memory_config = AggregatorConfig()

    mask_token_id = tokenizerMLM.mask_token_id
    layers = [-1]
    model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), layers, mask_token_id, memory_config, 1, None).to(
        device)
    model.emb_gen.load_state_dict(torch.load(path + "/pytorch_model.bin"))
    model.device = device
    model.firstLM.eval()
    model.secondLM.eval()

    model.eval()
    # train_examples = read_jsonl("train_relation.jsonl")
    with open("annotated_cot_for_relational.json", 'r') as fp:
        train_examples = json.load(fp)
    examples = read_jsonl("test_relation.jsonl")

    for k_shot in [1, 2, 4, 8]:
        outputs = []
        bad_examples = []
        print("{} shots...".format(k_shot))
        for i, ex in tqdm(enumerate(examples)):
            if i % 100 == 0:
                print("Processed {} examples".format(i))
            out_example = {}
            model.num_new_tokens = 1
            tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
            tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
            # try:
            out_text, text = run_example(model, tokenizerMLM, tokenizerTask, train_examples, ex, k_shot, let=let)
            out_example['input'] = text
            out_example['generation'] = out_text
            out_example['k_shot'] = k_shot
            # out_example['final_answer'] = ex['final_answer']
            out_example['original_example'] = ex

            outputs.append(out_example)
            # except:
            #     bad_examples.append(ex)

        with open("relational_test_outputs_emb_gen_old_let_{}_{}shot.json".format(let, k_shot), 'w') as fp:
            json.dump(outputs, fp)

        # with open("relational_error_examples_let_{}_{}shot.json".format(let, k_shot), 'w') as fp:
        #     json.dump(bad_examples, fp)
def main_multi(path, id, let=False, only_let=False, interleaved=False, mask_new_tokens=False):

    distributed_state = PartialState()
    device = distributed_state.device
    checkpt_name = path.split("/")[-1]
    tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
    tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
    nonces = list(tokenizerTask.get_added_vocab().keys())
    # tokenizerMLM.add_tokens(nonces)
    # tokenizerTask.add_tokens(nonces)
    firstLM = RobertaForMaskedLM.from_pretrained("roberta-large", low_cpu_mem_usage=True)
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                low_cpu_mem_usage=True, device_map=device)
    memory_config = AggregatorConfig()

    mask_token_id = tokenizerMLM.mask_token_id
    layers = [-1]
    model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), layers, mask_token_id, memory_config, 1, None).to(
        device)
    model.emb_gen.load_state_dict(torch.load(path + "/pytorch_model.bin"))
    model.device = device
    model.firstLM.eval()
    model.secondLM.eval()

    model.eval()
    # train_examples = read_jsonl("train_relation.jsonl")
    with open("annotated_cot_for_relational.json", 'r') as fp:
        train_examples = json.load(fp)
    examples = read_jsonl("test_relation.jsonl")
    with distributed_state.split_between_processes(examples) as partial_examples:
        for k_shot in [1, 2,4,8]:
            outputs = []
            bad_examples = []
            print("{} shots...".format(k_shot))
            for i, ex in tqdm(enumerate(partial_examples)):
                if i % 100 == 0:
                    print("Processed {} examples".format(i))
                out_example = {}
                model.num_new_tokens = 1
                tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
                tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
                # try:
                out_text, text = run_example(model, tokenizerMLM, tokenizerTask, train_examples, ex, k_shot, let=let, only_let=only_let, interleaved=interleaved, mask_new_tokens=mask_new_tokens)
                out_example['input'] = text
                out_example['generation'] = out_text
                out_example['k_shot'] = k_shot
                # out_example['final_answer'] = ex['final_answer']
                out_example['original_example'] = ex

                outputs.append(out_example)
                # except:
                #     bad_examples.append(ex)

            with open("relational_test_outputs_emb_gen_{}_let_{}_onlt_let_{}_interleaved_{}_{}shot_{}_id_{}_mask_{}.json".format(checkpt_name,let, only_let, interleaved, k_shot, distributed_state.process_index, id, mask_new_tokens), 'w') as fp:
                json.dump(outputs, fp)

        # with open("relational_error_examples_let_{}_{}shot.json".format(let, k_shot), 'w') as fp:
        #     json.dump(bad_examples, fp)

def prev_multi(path, id, let=False, only_let=False):

    distributed_state = PartialState()
    device = distributed_state.device
    tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
    tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
    nonces = list(tokenizerTask.get_added_vocab().keys())
    # tokenizerMLM.add_tokens(nonces)
    # tokenizerTask.add_tokens(nonces)
    firstLM = RobertaForMaskedLM.from_pretrained("roberta-large", low_cpu_mem_usage=True)
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                low_cpu_mem_usage=True, device_map=device)
    memory_config = AggregatorConfig()

    mask_token_id = tokenizerMLM.mask_token_id
    layers = [-1]
    model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), layers, mask_token_id, memory_config, 1, None).to(
        device)
    model.emb_gen.load_state_dict(torch.load(path + "/pytorch_model.bin"))
    model.device = device
    model.firstLM.eval()
    model.secondLM.eval()

    model.eval()
    # train_examples = read_jsonl("train_relation.jsonl")
    with open("annotated_cot_for_relational.json", 'r') as fp:
        train_examples = json.load(fp)
    examples = read_jsonl("test_relation.jsonl")
    with distributed_state.split_between_processes(examples) as partial_examples:
        for k_shot in [1, 2,4,8]:
            outputs = []
            bad_examples = []
            print("{} shots...".format(k_shot))
            for i, ex in tqdm(enumerate(partial_examples)):
                if i % 100 == 0:
                    print("Processed {} examples".format(i))
                out_example = {}
                model.num_new_tokens = 1
                tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
                tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
                # try:
                out_text, text = prev_run_example(model, tokenizerMLM, tokenizerTask, train_examples, ex, k_shot, let=let, only_let=only_let)
                out_example['input'] = text
                out_example['generation'] = out_text
                out_example['k_shot'] = k_shot
                # out_example['final_answer'] = ex['final_answer']
                out_example['original_example'] = ex

                outputs.append(out_example)
                # except:
                #     bad_examples.append(ex)

            with open("relational_test_outputs_emb_gen_prev_let_{}_{}shot_{}_id_{}.json".format(let, k_shot, distributed_state.process_index, id), 'w') as fp:
                json.dump(outputs, fp)

        # with open("relational_error_examples_let_{}_{}shot.json".format(let, k_shot), 'w') as fp:
        #     json.dump(bad_examples, fp)



def run_baseline(with_relation=True, let=False, only_let= False, var_names=False, remove_relation_for_test=False):
    device = "cuda"
    id = uuid.uuid4()
    model = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                             low_cpu_mem_usage=True).to(device)
    with open("annotated_cot_for_relational.json", 'r') as fp:
        train_examples = json.load(fp)

    examples = read_jsonl("test_relation.jsonl")
    model.eval()
    for k_shot in [2]:
        outputs = []
        bad_examples = []
        print("{} shots...".format(k_shot))
        for i, ex in tqdm(enumerate(examples)):
            out_example = {}
            tokenizer = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                       use_fast=False, legacy=True)
            # try:
            out_text, text = run_example_baseline(model, tokenizer, train_examples, ex, k_shot, with_relation, let=let, only_let=only_let, var_names=var_names, remove_relation_for_test = remove_relation_for_test)
            out_example['input'] = text
            out_example['generation'] = out_text
            out_example['k_shot'] = k_shot
            # out_example['final_answer'] = ex['final_answer']
            out_example['original_example'] = ex

            outputs.append(out_example)
            # except:
            #     bad_examples.append(ex)

        with open("relational_test_outputs_baseline_relation_{}_{}shot_{}_let_{}_only_let_{}_alphabet_{}.json".format(with_relation, k_shot, id, let, only_let, var_names), 'w') as fp:
            json.dump(outputs, fp)

        # with open("relational_error_examples_relation_{}_{}shot.json".format(with_relation, k_shot), 'w') as fp:
        #     json.dump(bad_examples, fp)
def get_arguments():
    parser = ArgumentParser()
    # parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--path", type=str)
    # parser.add_argument("--model", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--id", type=int)
    parser.add_argument("--only_let", action="store_true")
    parser.add_argument("--prev", action="store_true")
    parser.add_argument("--remove_relation_for_test", action="store_true")
    parser.add_argument("--interleaved", action="store_true")
    parser.add_argument("--mask_new_tokens", action="store_true")
    return parser


if __name__ == "__main__":
    # path="model_checkpoints/layers/no_mp/llama/input_and_output/filtered/pile/layernorm/roberta-large/1_layers/last_1/32_batch_size/mean_agg/1_examples/lr_0.001/weight_decay_0.1/with_negatives_and_regression/distillation_weight_0.05_temp_3/output_embedding_cosine/checkpoints/checkpoint_4_8500"
    # path = "model_checkpoints/layers/no_mp/llama/input_and_output/filtered/redone_pile/layernorm/roberta-large/1_layers/last_1/32_batch_size/mean_agg/1_examples/lr_0.001/weight_decay_0.1/with_negatives_and_regression/distillation_weight_0.05_temp_3/output_embedding_cosine/checkpoints/checkpoint_2_9000"
    # main(path, let=True)
    args = get_arguments().parse_args()
    if args.model == "vanilla":
        run_vanilla(remove_relation_for_test=args.remove_relation_for_test)
    if args.model == "baseline_relation":
        run_baseline(with_relation=True, let=False, remove_relation_for_test=args.remove_relation_for_test)
    if args.model == "let_baseline":
        run_baseline(with_relation=False, let=True, only_let=args.only_let)
    if args.model == "emb_gen":
        path = "model_checkpoints/layers/no_mp/llama/input_and_output/filtered/redone_pile/layernorm/roberta-large/1_layers/last_1/32_batch_size/mean_agg/1_examples/lr_0.001/weight_decay_0.1/with_negatives_and_regression/distillation_weight_0.05_temp_3/output_embedding_cosine/checkpoints/checkpoint_7_28000"
        if args.prev:
            prev_multi(path, id=args.id, let=True, only_let=args.only_let)
        else:
            main_multi(path, id=args.id, let=True, only_let=args.only_let, interleaved=args.interleaved, mask_new_tokens=args.mask_new_tokens)
    if args.model == "alphabet":
        run_baseline(with_relation=False, let=True, only_let=args.only_let, var_names=True)
    # print("running with relation=True")
    # run_baseline(True)
    # print("running with relation=False")
    # run_baseline(False)
