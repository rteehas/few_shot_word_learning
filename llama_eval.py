import torch
import numpy as np
import itertools

def prepare_for_top_1_selection(ex):
    multi_blank_vals = ["(i)", "(ii)", "(iii)"]
    question = ex["QUESTION"]
    answers = ex["ANSWERS"]
    task_seqs = []
    if "_____" in question:
        answers = answers[0]
        for answer in answers:
            task_seqs.append(question.replace("_____", answer))

    elif "(i)" in question:
        to_replace = [i for i in multi_blank_vals if i in question]
        assert len(to_replace) == len(ex["LABELS"])

        combinations = list(itertools.product(*answers))

        for comb in combinations:
            tmp_q = question
            for s, a in zip(to_replace, comb):
                tmp_q = tmp_q.replace(s, a)

            task_seqs.append(tmp_q)
    else:
        raise NotImplementedError

    labels = ex["LABELS"]

    seq_label = []
    for seq in task_seqs:
        if not all(a in seq for a in labels):
            seq_label.append(0)
        else:
            seq_label.append(1)

    return task_seqs, seq_label


def prepare_for_top_2_selection(ex):
    question = ex["QUESTION"]
    answers = ex["ANSWERS"]
    task_seqs = []
    assert "_____" in question
    answers = answers[0]
    for answer in answers:
        task_seqs.append(question.replace("_____", answer))

    labels = ex["LABELS"]

    seq_label = []
    for seq in task_seqs:
        if not any(a in seq for a in labels):
            seq_label.append(0)
        else:
            seq_label.append(1)

    return task_seqs, seq_label

def get_sentence_probs(model, tokenizer, sequences):
    probs = []
    for seq in sequences:
        toks = tokenizer(seq, return_tensors="pt").to(model.device)
        labels = toks['input_ids'].clone()
        out = model(input_ids=toks['input_ids'], attention_mask=toks['attention_mask'], labels=labels)
        probs.append(-out.loss.item())
    return probs


def evaluate_type_1(probs, labels):
    probs = np.array(probs)
    max_prob = np.argmax(probs)
    lab_id = labels.index(1)

    return max_prob == lab_id


def evaluate_type_2(probs, labels):
    probs = np.array(probs)
    idx = np.argsort(probs, axis=0)[-2:]
    lab_ids = [i for i, v in enumerate(labels) if v == 1]

    return set(idx) == set(lab_ids)


def evaluate_baseline_example(model, tokenizer, ex):
    if ex["ANSWER_TYPE"] == "top_1":
        seqs, labels = prepare_for_top_1_selection(ex)
    elif ex["ANSWER_TYPE"] == "top_2":
        seqs, labels = prepare_for_top_2_selection(ex)
    else:
        raise NotImplementedError

    probs = get_sentence_probs(model, tokenizer, seqs)

    if ex["ANSWER_TYPE"] == "top_1":
        return evaluate_type_1(probs, labels)
    elif ex["ANSWER_TYPE"] == "top_2":
        return evaluate_type_2(probs, labels)


def prepare_type_1_fewshot(ex, sent_dict, k, with_definition=False, defs=None):
    sentence_template = "You are given a set of example sentences for a new term or terms and must assess a sentence using it.\nWord: {}\nExamples: {}\nSentence: {}"
    definition_template = "You are given a set of example sentences and a definition for a new term or terms and must assess a sentence using it.\nWord: {}\nDefinition: {}.\nExamples: {}\nSentence: {}"
    base_seqs, labels = prepare_for_top_1_selection(ex)
    multi_blank_vals = ["(i)", "(ii)", "(iii)"]
    question = ex["QUESTION"]
    answers = ex["ANSWERS"]
    task_seqs = []
    if "_____" in question:
        answers = answers[0]


    elif "(i)" in question:
        answers = list(itertools.product(*answers))
    seqs = []
    #     print(answers)
    for w, s in zip(answers, base_seqs):
        if type(w) == str:
            nonce = "<{}_new>".format(w)
            #             print(w)
            samples = np.random.choice(sent_dict[w], size=k)
            samples = [re.sub(r"\b({})\b".format(w), nonce, sentence, flags=re.I) for sentence in samples]
            example_string = " \n".join(samples)
        else:
            raise NotImplementedError("Answer must be string")
        #             samples = []
        #             for val in w:
        #                 samples.append(np.random.choice(sent_dict[w]), size=k)

        #             examples = [" ".join(sample) for sample in samples]
        #             example_string = "\n".join(examples)
        new_s = re.sub(r"\b({})\b".format(w), nonce, s, flags=re.I)
        if with_definition and defs is not None:
            definition = defs[w]
            seq = definition_template.format(nonce, definition, example_string, new_s)
        else:
            seq = sentence_template.format(nonce, example_string, new_s)
        seqs.append(seq)

    return seqs, labels


def prepare_for_type_2_fewshot(ex, sent_dict, k, with_definition=False, defs=None):
    sentence_template = "You are given a set of example sentences for a new term or terms and must assess a sentence using it.\nWord: {}\nExamples: {}\nSentence: {}"
    definition_template = "You are given a set of example sentences and a definition for a new term or terms and must assess a sentence using it.\nWord: {}\nDefinition: {}.\nExamples: {}\nSentence: {}"

    base_seqs, labels = prepare_for_top_2_selection(ex)
    answers = ex["ANSWERS"][0]
    seqs = []
    for w, s in zip(answers, base_seqs):
        nonce = "<{}_new>".format(w)
        samples = np.random.choice(sent_dict[w], size=k)
        samples = [re.sub(r"\b({})\b".format(w), nonce, sentence, flags=re.I) for sentence in samples]
        example_string = " \n".join(samples)

        new_s = re.sub(r"\b({})\b".format(w), nonce, s, flags=re.I)
        if with_definition and defs is not None:
            definition = defs[w]
            seq = definition_template.format(nonce, definition, example_string, new_s)
        else:
            seq = sentence_template.format(nonce, example_string, new_s)
        seqs.append(seq)

    return seqs, labels