import torch
import numpy as np
import itertools
import re

from torch.nn import CrossEntropyLoss


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

def evaluate_baseline_example_fewshot(model, tokenizer, ex, sents, k, with_definition=False, defs=None):
    if ex["ANSWER_TYPE"] == "top_1":
        seqs, labels, base_seqs = prepare_type_1_fewshot(ex, sents, k, with_definition, defs)
    elif ex["ANSWER_TYPE"] == "top_2":
        seqs, labels, base_seqs = prepare_for_type_2_fewshot(ex, sents, k, with_definition, defs)
    else:
        raise NotImplementedError

    probs = get_sentence_probs(model, tokenizer, seqs, base_seqs)
    # print(probs)
    if ex["ANSWER_TYPE"] == "top_1":
        return evaluate_type_1(probs, labels)
    elif ex["ANSWER_TYPE"] == "top_2":
        return evaluate_type_2(probs, labels)


def prepare_type_1_fewshot(ex, sent_dict, k, with_definition=False, defs=None):
    sentence_stem = "You are given a set of example sentences for a new term or terms and must assess a sentence using it.\n"
    definition_stem = "You are given a set of example sentences and a definition for a new term or terms and must assess a sentence using it.\n"
    sentence_template = "Word: {}\nExamples: {}\n"
    definition_template = "Word: {}\nDefinition: {}.\nExamples: {}\n"
    seq_template = "Sentence: {}"
    nonce_template = "<{}_new>"

    base_seqs, labels = prepare_for_top_1_selection(ex)
    multi_blank_vals = ["(i)", "(ii)", "(iii)"]
    question = ex["QUESTION"]
    answers = ex["ANSWERS"]
    task_seqs = []
    if "_____" in question:
        answers = answers[0]


    elif "(i)" in question:
        answer_choices = list(itertools.chain(*answers))
        answers = list(itertools.product(*answers))
        answer_samples = {}
        for w in answer_choices:
            answer_samples[w] = np.random.choice(sent_dict[w], size=k, replace=False)

    seqs = []
    #     print(answers)
    question_seqs = []
    for w, s in zip(answers, base_seqs):
        if type(w) == str:
            nonce = "<{}_new>".format(w.lower())
            #             print(w)
            samples = np.random.choice([s for s in sent_dict[w] if re.search(r"\b({})\b".format(w), s, flags=re.I) is not None], size=k, replace=False)
            samples = [re.sub(r"\b({})\b".format(w), nonce, sentence, flags=re.I) for sentence in samples]
            examples = [" \n".join(samples)]
        else:
            examples = [re.sub(r"\b({})\b".format(w), nonce, " \n".join(answer_samples[v]), flags=re.I) for v in w]
        #             examples = [" \n".join(sample) for sample in samples]

        if with_definition and defs is not None:
            if type(w) == str:
                nonce = nonce_template.format(w.lower())
                definition = defs[w]
                formatted_examples_with_definition = [definition_template.format(nonce, definition, ex) for ex in
                                                      examples]
            else:
                formatted_examples_with_definition = []
                for i, v in enumerate(w):
                    nonce = nonce_template.format(v.lower())
                    definition = defs[v]
                    formatted_example = definition_template.format(nonce, definition, examples[i])
                    formatted_examples_with_definition.append(formatted_example)

            seq_minus_sentence = definition_stem + "".join(formatted_examples_with_definition)
        else:
            if type(w) == str:
                nonce = nonce_template.format(w.lower())
                formatted_examples = [sentence_template.format(nonce, ex) for ex in examples]

            else:
                formatted_examples = []
                for i, v in enumerate(w):
                    nonce = nonce_template.format(v.lower())
                    formatted_example = sentence_template.format(nonce, examples[i])
                    formatted_examples.append(formatted_example)

            seq_minus_sentence = sentence_stem + "".join(formatted_examples)

        if type(w) == str:
            nonce = nonce_template.format(w.lower())
            new_s = re.sub(r"\b({})\b".format(w), nonce, s, flags=re.I)

        else:
            new_s = s
            for v in w:
                new_s = re.sub(r"\b({})\b".format(v), nonce_template.format(v), new_s)
        seq = seq_minus_sentence + seq_template.format(new_s)
        question_seqs.append(new_s)
        seqs.append(seq)

    return seqs, labels, question_seqs


def prepare_for_type_2_fewshot(ex, sent_dict, k, with_definition=False, defs=None):
    sentence_template = "You are given a set of example sentences for a new term or terms and must assess a sentence using it.\nWord: {}\nExamples: {}\nSentence: {}"
    definition_template = "You are given a set of example sentences and a definition for a new term or terms and must assess a sentence using it.\nWord: {}\nDefinition: {}.\nExamples: {}\nSentence: {}"

    base_seqs, labels = prepare_for_top_2_selection(ex)
    answers = ex["ANSWERS"][0]
    seqs = []
    question_seqs = []
    for w, s in zip(answers, base_seqs):
        nonce = "<{}_new>".format(w.lower())
        samples = np.random.choice([s for s in sent_dict[w] if re.search(r"\b({})\b".format(w), s, flags=re.I) is not None], size=k)
        samples = [re.sub(r"\b({})\b".format(w), nonce, sentence, flags=re.I) for sentence in samples]
        example_string = " \n".join(samples)

        new_s = re.sub(r"\b({})\b".format(w), nonce, s, flags=re.I)
        if with_definition and defs is not None:
            definition = defs[w]
            seq = definition_template.format(nonce, definition, example_string, new_s)
        else:
            seq = sentence_template.format(nonce, example_string, new_s)
        seqs.append(seq)
        question_seqs.append(new_s)

    return seqs, labels, question_seqs

def prepare_emb_gen_batch(ex, sent_dict, k):

    if ex["ANSWER_TYPE"] == "top_1":
        question = ex["QUESTION"]
        answers = ex["ANSWERS"]
        if "_____" in question:
            answers = answers[0]


        elif "(i)" in question:
            # answers = list(itertools.product(*answers))
            raise NotImplementedError

        seqs, labels = prepare_for_top_1_selection(ex)

    elif ex["ANSWER_TYPE"] == "top_2":
        seqs, labels = prepare_for_top_2_selection(ex)
        answers = ex["ANSWERS"][0]
    else:
        raise NotImplementedError

    task_seqs = []
    task_samples = []
    for w, task_s in zip(answers, seqs):
        if type(w) == str:
            nonce = "<{}_new>".format(w.lower())
            samples = np.random.choice([s for s in sent_dict[w] if re.search(r"\b({})\b".format(w), s, flags=re.I) is not None], size=k, replace=False)
            # samples = [s for s in samples if re.search(r"\b({})\b".format(w), s, flags=re.I) is not None]
            samples = [re.sub(r"\b({})\b".format(w), nonce, s, flags=re.I) for s in samples]
            # print("Samples for {}".format(w), samples)
            # new_samples = []
            # for s in samples:
            #     if w in s:
            #         new_samples.append(s.replace(w, nonce))
            #     elif w.capitalize() in s:
            #         new_samples.append(w.capitalize(), nonce)
            # # samples = [s.replace(w, nonce) if w in s else s.replace(w.capitalize(), nonce) if w.capitalize() in s for s in samples]
            # samples = new_samples
            task_samples.append(samples)
            task_seqs.append(re.sub(r"\b({})\b".format(w), nonce, task_s, flags=re.I))
        else:
            raise NotImplementedError

    return task_samples, task_seqs, labels


@torch.no_grad()
def get_sentence_probs_emb_gen(model, tokenizerMLM, tokenizerTask, contexts, seqs):
    probs = []
    for i,seq in enumerate(seqs):
        context = tokenizerMLM(contexts[i], padding=True, truncation=True, max_length=256, return_tensors='pt')
        # print(context)
        toks = tokenizerTask(seq, truncation=True, max_length=256, return_tensors="pt").to(model.device)
        labels = toks['input_ids'].clone()
        batch = {
            "contexts": [context],
            "input_ids": toks['input_ids'],
            "attention_mask": toks['attention_mask'],
            'labels': labels
        }
        out = model(batch)
        probs.append(-out.loss.item())
    return probs

@torch.no_grad()
def get_sentence_probs(model, tokenizer, sequences, base_seqs):
    probs = []
    ce = CrossEntropyLoss()
    for seq,base in zip(sequences, base_seqs):
        toks = tokenizer(seq, return_tensors="pt").to(model.device)
        question_toks = tokenizer(base)
        answer_length = len(question_toks['input_ids']) - 1 # for bos token
        # print(question_toks, answer_length)
        labels = toks['input_ids'].clone()
        answer_labels = labels[:, -answer_length:]
        # print(answer_labels)
        out = model(input_ids=toks['input_ids'], attention_mask=toks['attention_mask'], labels=labels)
        answer_logits = out.logits[:,-answer_length:, :]
        shift_logits = answer_logits[..., :-1, :].contiguous()
        shift_labels = answer_labels[..., 1:].contiguous()
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = ce(shift_logits, shift_labels)
        probs.append(-loss.item())
    return probs


def filter_gre(sents, ex):
    question = ex["QUESTION"]
    answers = ex["ANSWERS"]
    if "_____" in question:
        answers = answers[0]

    elif "(i)" in question:
        answers = list(itertools.chain(*answers))

    is_valid = True

    for a in answers:
        if a not in sents or len(sents[a]) == 0:
            is_valid = False

    return is_valid

def evaluate_emb_gen(model, tokenizerMLM, tokenizerTask, ex, sents, k):
    samples, seqs, labels = prepare_emb_gen_batch(ex, sents, k)
    probs = get_sentence_probs_emb_gen(model, tokenizerMLM, tokenizerTask, samples, seqs)
    if ex["ANSWER_TYPE"] == "top_1":
        return evaluate_type_1(probs, labels)
    elif ex["ANSWER_TYPE"] == "top_2":
        return evaluate_type_2(probs, labels)





