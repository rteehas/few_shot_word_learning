import re
import json
from collections import OrderedDict
import numpy as np


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


def process_answer(ex):
    explanations, equations = get_explanations_and_equations(ex)
    #     print(explanations, equations)
    mapping = None
    answer = ex['answer']
    processed = []
    for i, expl in enumerate(explanations):
        processed_explanation = process_explanation(expl, mapping)

        eq_with_nonces = "{} {}".format(processed_explanation[1], equations[i])
        answer = answer.replace(equations[i], eq_with_nonces)
        mapping = processed_explanation[2]
        processed.append((processed_explanation[0], processed_explanation[1]))

    return answer, mapping, processed

def remove_answer_for_eval(answer):
    delimiter = "Final answer:"
    return answer.split(delimiter)[0] + delimiter

def construct_context(mapping, example_list):
    context_mapping = OrderedDict()
    for nonce in mapping.values():
        print(nonce)
        nonce_context = []
        for example in example_list:
            for sentence in example:
                if nonce in sentence:
                    nonce_context.append(sentence)
        context_mapping[nonce] = nonce_context
    return list(context_mapping.items())


def create_example(ex, mapping=None, use_one_example=False):
    answer, mapping, processed = process_answer(ex, mapping=mapping)
    example_list = [p[0] for p in processed]
    context = construct_context(mapping, example_list)

    if use_one_example:
        context = [[c[0]] for c in context]

    return context, answer, mapping

def process_for_eval(train_set, test_example, k_shot=0, use_one_example=False):
    mapping = None
    if k_shot > 0:
        sampled_k_shot_examples = np.random.choice(train_set, size=k_shot, replace=False)
        contexts = []
        answers = []
        for ex in sampled_k_shot_examples:
            context, answer, mapping = create_example(ex, mapping=mapping, use_one_example=use_one_example)
            contexts += context
            answers.append(answer)
    #             print(contexts)

    test_context, test_answer, final_mapping = create_example(test_example, mapping=mapping,
                                                              use_one_example=use_one_example)
    truncated_answer = remove_answer_for_eval(test_answer)
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
    if len(tokenizerMLM.get_added_tokens()) < tokens_needed_for_task:
        new_tokens = ["<nonce{}>".format(i) for i in range(1,tokens_needed_for_task)]
        tokenizerMLM.add_tokens(new_tokens)
    if len(tokenizerTask.get_added_tokens()) < tokens_needed_for_task:
        new_tokens = ["<nonce{}>".format(i) for i in range(1,tokens_needed_for_task)]
        tokenizerMLM.add_tokens(new_tokens)

    if model.num_new_tokens < tokens_needed_for_task:
        model.num_new_tokens = tokens_needed_for_task