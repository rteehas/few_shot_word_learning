import re
import json
from collections import OrderedDict
import numpy as np
from train_with_llama import *
from transformers import RobertaForMaskedLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, \
    get_linear_schedule_with_warmup, AdamW, DataCollatorForLanguageModeling, AutoConfig, T5EncoderModel
from tqdm import tqdm
import json

def read_jsonl(path: str):
    """Read JSON file. Copied from gsm.py"""
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

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
        final_relation = final_relation.replace(comp, nonce)
        masked_examples.append(explanation.replace(comp, nonce))
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
        processed.append((processed_explanation[0],processed_explanation[1]))

    return answer, mapping, processed

def remove_answer_for_eval(answer):
    delimiter = "Final answer:"
    return answer.split(delimiter)[0] + delimiter

def construct_context(mapping, example_list):
    context_mapping = OrderedDict()
    for nonce in mapping.values():
#         print(nonce)
        nonce_context = []
        for example in example_list:
            for sentence in example:
                if nonce in sentence:
                    nonce_context.append(sentence)
        context_mapping[nonce] = nonce_context
    return list(context_mapping.values())


def create_example(ex, mapping=None, use_one_example=False, no_text=False, let=False):
    answer, mapping, processed = process_answer(ex, mapping=mapping, no_text=no_text)
    if let:
        first_let_str = "Let {} = {}"
        second_let_str = first_let_str.lower()

        lets = []
        for i, (k, v) in enumerate(mapping.items()):
            if i == 0:
                lets.append(first_let_str.format(k, v) + ",")
            elif i < len(list(mapping.keys())) - 1:
                lets.append(" " + second_let_str.format(k, v) + ",")
            else:
                lets.append(" " + second_let_str.format(k, v) + ".\n")

        preamble = " ".join(lets)
        answer = preamble + answer

    example_list = [p[0] for p in processed]
    context = construct_context(mapping, example_list)

    if use_one_example:
        context = [[c[0]] for c in context]

    return context, answer, mapping


def process_for_eval(train_set, test_example, k_shot=0, use_one_example=False, no_text=False, let=False):
    mapping = None
    if k_shot > 0:
        sampled_k_shot_examples = np.random.choice(train_set, size=k_shot, replace=False)
        contexts = []
        answers = []
        for ex in sampled_k_shot_examples:
            context, answer, mapping = create_example(ex, mapping=mapping, use_one_example=use_one_example, no_text=no_text, let=let)
            answer = "{}\n{}".format(ex['question'], answer)
            answer = answer + "Final answer: {}".format(ex['final_answer'])
            contexts += context
            answers.append(answer)
    #             print(contexts)

    test_context, test_answer, final_mapping = create_example(test_example, mapping=mapping,
                                                              use_one_example=use_one_example, no_text=no_text, let=let)

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
        new_tokens = ["<nonce{}>".format(i) for i in range(1,tokens_needed_for_task)]
        tokenizerMLM.add_tokens(new_tokens)
    if len(tokenizerTask.get_added_vocab()) < tokens_needed_for_task:
        new_tokens = ["<nonce{}>".format(i) for i in range(1,tokens_needed_for_task)]
        tokenizerMLM.add_tokens(new_tokens)

    if model.num_new_tokens < tokens_needed_for_task:
        model.num_new_tokens = tokens_needed_for_task

def verify_or_fix_baseline_num_tokens(model, tokenizer, context):
    tokens_needed_for_task = len(context)
    if len(tokenizer.get_added_vocab()) < tokens_needed_for_task:
        new_tokens = ["<nonce{}>".format(i) for i in range(1,tokens_needed_for_task)]
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))

def run_example(model, tokenizerMLM, tokenizerTask, train_examples, ex, k_shot, let=False):
    contexts, text, ans = process_for_eval(train_examples,ex, k_shot, use_one_example=False, no_text=True, let=let)
    verify_or_fix_num_tokens(model, tokenizerMLM, tokenizerTask, contexts)

    ctx = [tokenizerMLM(c, padding="longest", truncation=True, return_tensors='pt').to(model.device) for c in contexts]
    target_input = tokenizerTask(text, return_tensors='pt').to(model.device)
    gen_out = generate_multi(model, ctx, target_input['input_ids'], target_input['attention_mask'], 30, mask_new_tokens=True, top_k=10)
    out_text = tokenizerTask.decode(gen_out[0])
    return out_text, text

def run_example_baseline(model, tokenizer, train_examples, ex, k_shot, with_relation):
    if not with_relation:
        contexts, text, ans = process_for_eval(train_examples, ex, k_shot, use_one_example=False, no_text=True)
        verify_or_fix_baseline_num_tokens(model, tokenizer, contexts)
    else:
        text, ans = process_for_baseline_eval(train_examples, ex, k_shot)

    target_input = tokenizer(text, return_tensors='pt').to(model.device)
    gen_out = model.generate(**target_input, use_cache=True, top_k=10, max_new_tokens=50)
    out_text = tokenizer.decode(gen_out[0])
    return out_text, text

def process_baseline_answer_with_relation(ex):
    explanations, equations = get_explanations_and_equations(ex)
    answer = ex['answer']
    for i, expl in enumerate(explanations):
        eq = equations[i]
        eq_with_relation = "{} {}".format(expl, eq)
        answer = answer.replace(eq, eq_with_relation)
    return answer

def process_for_baseline_eval(train_set, test_example, k_shot=0):
    if k_shot > 0:
        sampled_k_shot_examples = np.random.choice(train_set, size=k_shot, replace=False)
        answers = []
        for ex in sampled_k_shot_examples:
            answer = process_baseline_answer_with_relation(ex)
            answers.append(answer)
    test_answer = process_baseline_answer_with_relation(test_example)
    test_expl, test_eq = get_explanations_and_equations(test_example)
    truncated_answer = test_answer.split(test_eq[-1])[0]
    if k_shot > 0:
        answer_text = "\n".join(answers)
        answer_text = answer_text + "\n" + truncated_answer
        return answer_text, test_example['final_answer']
    else:
        return truncated_answer, test_example['final_answer']

def main(path, let=False):
    device = "cuda"
    tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
    tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
    nonces = list(tokenizerTask.get_added_vocab().keys())
    # tokenizerMLM.add_tokens(nonces)
    # tokenizerTask.add_tokens(nonces)
    firstLM = RobertaForMaskedLM.from_pretrained("roberta-large", low_cpu_mem_usage=True)
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", low_cpu_mem_usage=True)
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

    for k_shot in [1,2,3,4]:
        outputs = []
        bad_examples = []
        print("{} shots...".format(k_shot))
        for i, ex in enumerate(examples):
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

        with open("relational_test_outputs_emb_gen_let_{}_{}shot.json".format(let, k_shot), 'w') as fp:
            json.dump(outputs, fp)

        # with open("relational_error_examples_let_{}_{}shot.json".format(let, k_shot), 'w') as fp:
        #     json.dump(bad_examples, fp)

def run_baseline(with_relation=True):
    device = "cuda"
    model = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                low_cpu_mem_usage=True).to(device)
    with open("annotated_cot_for_relational.json", 'r') as fp:
        train_examples = json.load(fp)

    examples = read_jsonl("test_relation.jsonl")
    model.eval()
    for k_shot in [1, 2, 3, 4]:
        outputs = []
        bad_examples = []
        print("{} shots...".format(k_shot))
        for i, ex in tqdm(enumerate(examples)):
            out_example = {}
            tokenizer = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                       use_fast=False, legacy=True)
            # try:
            out_text, text = run_example_baseline(model, tokenizer, train_examples, ex, k_shot, with_relation)
            out_example['input'] = text
            out_example['generation'] = out_text
            out_example['k_shot'] = k_shot
            # out_example['final_answer'] = ex['final_answer']
            out_example['original_example'] = ex

            outputs.append(out_example)
            # except:
            #     bad_examples.append(ex)

        with open("relational_test_outputs_baseline_relation_{}_{}shot.json".format(with_relation, k_shot), 'w') as fp:
            json.dump(outputs, fp)

        # with open("relational_error_examples_relation_{}_{}shot.json".format(with_relation, k_shot), 'w') as fp:
        #     json.dump(bad_examples, fp)


if __name__ == "__main__":
    path="model_checkpoints/layers/no_mp/llama/input_and_output/filtered/pile/layernorm/roberta-large/1_layers/last_1/32_batch_size/mean_agg/1_examples/lr_0.001/weight_decay_0.1/with_negatives_and_regression/distillation_weight_0.05_temp_3/output_embedding_cosine/checkpoints/checkpoint_4_8500"
    # main(path, let=True)
    print("running with relation=True")
    run_baseline(True)
    print("running with relation=False")
    run_baseline(False)


