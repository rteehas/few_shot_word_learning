from copy import deepcopy

import torch
import numpy as np
import itertools
import re
from train_with_llama import *
from torch.nn import CrossEntropyLoss

from twitter_eval import get_sentence_probs_agnostic
from w2v_baselines import make_hice_batch, make_w2v_batch

def prepare_for_t5(seq, nonce):
    t5_format = "<extra_id_{}>"
    ct = 0
    tmp_seq = seq
    while nonce in tmp_seq:
        tmp_seq = tmp_seq.replace(nonce, t5_format.format(ct), 1)
        ct += 1

    return tmp_seq


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

def evaluate_baseline_example_fewshot(model, tokenizer, ex, sents, with_definition=False, defs=None, tuning=False, max_steps=2, with_prompt=True, lr=1e-3):
    if not tuning and not with_prompt:
        raise NotImplementedError
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
    # print(samples)
    # print(seqs)
    # print(base_seqs)
    tokenizer.pad_token = tokenizer.unk_token
    orig_input_embeds = model.get_input_embeddings().weight.clone()
    orig_output_embeds = model.get_output_embeddings().weight.clone()
    if tuning:
        outputs = []
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_input_embeddings().parameters():
            param.requires_grad = True
        for param in model.get_output_embeddings().parameters():
            param.requires_grad = True

        new_tok_indices = [v for k,v in tokenizer.get_added_vocab().items()]
        zero_grad_indices = torch.arange(0, len(tokenizer)) != any(new_tok_indices)

        total_probs = []
        if with_prompt:
            for sample, seq, base_seq in zip(samples, seqs, base_seqs):
                opt = AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-3)
                input = tokenizer(sample, padding='longest', return_tensors='pt')
                per_step_probs = []
                for step in range(max_steps):
                    model.train()
                    model.zero_grad()
                    opt.zero_grad()
                    output = model(input_ids=input['input_ids'].to(model.device),
                                   attention_mask=input['attention_mask'].to(model.device),
                                   labels=input['input_ids'].clone().to(model.device))

                    loss = output.loss
                    loss.backward()
                    opt.step()

                    model.get_input_embeddings().weight.grad[zero_grad_indices] = 0.
                    model.get_output_embeddings().weight.grad[zero_grad_indices] = 0.

                    with torch.no_grad():
                        model.eval()
                        prob = get_sentence_probs(model, tokenizer, [seq], [base_seq])
                        # print(prob)
                        per_step_probs.append(prob[0])

                total_probs.append(per_step_probs)

                    # reset embeddings
                model.get_input_embeddings().weight = torch.nn.Parameter(orig_input_embeds)
                model.get_output_embeddings().weight = torch.nn.Parameter(orig_output_embeds)

        else:
            for sample, seq in zip(samples, seqs):
                opt = AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr)
                input = tokenizer(sample, padding='longest', return_tensors='pt')
                per_step_probs = []
                for step in range(max_steps):
                    model.train()
                    model.zero_grad()
                    opt.zero_grad()
                    output = model(input_ids=input['input_ids'].to(model.device),
                                   attention_mask=input['attention_mask'].to(model.device),
                                   labels=input['input_ids'].clone().to(model.device))

                    loss = output.loss
                    loss.backward()
                    opt.step()

                    model.get_input_embeddings().weight.grad[zero_grad_indices] = 0.
                    model.get_output_embeddings().weight.grad[zero_grad_indices] = 0.

                    with torch.no_grad():
                        model.eval()
                        toks = tokenizer(seq, return_tensors="pt").to(model.device)
                        label = toks['input_ids'].clone()
                        out = model(input_ids = toks['input_ids'], attention_mask=toks['attention_mask'], labels=label)
                        # prob = get_sentence_probs(model, tokenizer, [seq], [base_seq])
                        prob = -out.loss.item()
                        per_step_probs.append(prob)

                total_probs.append(per_step_probs)

                #reset embeddings
                model.get_input_embeddings().weight = torch.nn.Parameter(orig_input_embeds)
                model.get_output_embeddings().weight = torch.nn.Parameter(orig_output_embeds)

        seq_probs_by_step = [[seq_prob[step] for seq_prob in total_probs] for step in range(max_steps)]
        # print(seq_probs_by_step)

        example_outputs_by_step = []
        for probs in seq_probs_by_step:
            if ex["ANSWER_TYPE"] == "top_1":
                example_outputs_by_step.append(evaluate_type_1(probs, labels))
            elif ex["ANSWER_TYPE"] == "top_2":
                example_outputs_by_step.append(evaluate_type_2(probs, labels))

        return example_outputs_by_step
    else:
        with torch.no_grad():
            probs = get_sentence_probs(model, tokenizer, seqs, base_seqs)
            # print(probs)
            if ex["ANSWER_TYPE"] == "top_1":
                return evaluate_type_1(probs, labels)
            elif ex["ANSWER_TYPE"] == "top_2":
                return evaluate_type_2(probs, labels)


def prepare_type_1_fewshot(ex, sent_dict,with_definition=False, defs=None, with_prompt=True):
    # sentence_stem = "You are given a set of example sentences for a new term or terms and must assess a sentence using it.\n"
    # definition_stem = "You are given a set of example sentences and a definition for a new term or terms and must assess a sentence using it.\n"
    if with_prompt:
        sentence_template = "Here are some sentences for a new word \"{}\":\n{}"

    nonce = "<nonce>"
    base_seqs, labels = prepare_for_top_1_selection(ex)
    multi_blank_vals = ["(i)", "(ii)", "(iii)"]
    question = ex["QUESTION"]
    answers = ex["ANSWERS"]
    task_seqs = []
    if "_____" in question:
        answers = answers[0]


    elif "(i)" in question:
        raise NotImplementedError
        # answer_choices = list(itertools.chain(*answers))
        # answers = list(itertools.product(*answers))
        # answer_samples = {}
        # for w in answer_choices:
        #     answer_samples[w] = np.random.choice(sent_dict[w], size=k, replace=False)

    seqs = []
    #     print(answers)
    question_seqs = []
    final_samples = []
    for w, s in zip(answers, base_seqs):
        if type(w) == str:
            # nonce = "<nonce>"
            #             print(w)
            samples = sent_dict[w]
            samples = [re.sub(r"\b({})\b".format(w), nonce, sentence, flags=re.I) for sentence in samples]
            # examples = [" \n".join(samples)]
            final_samples.append(samples)

        else:
            raise NotImplementedError
            # examples = [re.sub(r"\b({})\b".format(w), nonce, " \n".join(answer_samples[v]), flags=re.I) for v in w]
        #             examples = [" \n".join(sample) for sample in samples]


        # if with_definition and defs is not None:
        #     if type(w) == str:
        #         # nonce = "<nonce>"
        #         definition = defs[w]
        #         def_s = "The word {} is defined as {}".format(nonce, definition)
        #         formatted_examples_with_definition = []
        #         for example in examples:
        #             new_example = example + "\n" + def_s
        #             formatted_examples_with_definition.append(sentence_template.format(nonce, new_example))
        #
        #         final_samples.append(samples + [def_s])
        #     else:
        #         formatted_examples_with_definition = []
        #         for i, v in enumerate(w):
        #             # nonce = nonce_template.format(v.lower())
        #             definition = defs[v]
        #             def_s = "The word {} is defined as {}".format(nonce, definition)
        #             new_example = examples[i] + "\n" + def_s
        #             formatted_example = sentence_template.format(nonce, "\n".join(new_example))
        #             formatted_examples_with_definition.append(formatted_example)
        #             converted_ans = [re.sub(r"\b({})\b".format(w), nonce, tmp_sample, flags=re.I) for tmp_sample in answer_samples[v]]
        #             final_samples.append(converted_ans + [def_s])
        #
        #
        #     seq_minus_sentence = sentence_template.format(nonce, "".join(formatted_examples_with_definition))
        # else:
        # if type(w) == str:
        #     # nonce = nonce_template.format(w.lower())
        #     formatted_examples = [sentence_template.format(nonce, ex) for ex in examples]
        #     final_samples.append(samples)

        # else:
            # formatted_examples = []
            # for i, v in enumerate(w):
            #     # nonce = nonce_template.format(v.lower())
            #     formatted_example = sentence_template.format(nonce, examples[i])
            #     formatted_examples.append(formatted_example)
            #     converted_ans = [re.sub(r"\b({})\b".format(w), nonce, tmp_sample, flags=re.I) for tmp_sample in
            #                      answer_samples[v]]
            #     final_samples.append(converted_ans)

        # seq_minus_sentence = sentence_template.format(nonce,"".join(formatted_examples))

        if type(w) == str:
            # nonce = nonce_template.format(w.lower())
            new_s = re.sub(r"\b({})\b".format(w), nonce, s, flags=re.I)

        else:
            raise NotImplementedError
            # new_s = s
            # for v in w:
            #     new_s = re.sub(r"\b({})\b".format(v), nonce_template.format(v), new_s)
        samples_with_task = samples + [new_s]
        if with_prompt:
            seq = sentence_template.format("<nonce>", "\n".join(samples_with_task))
            seqs.append(seq)
        question_seqs.append(new_s)

    if with_prompt:
        return seqs, labels, question_seqs, final_samples
    else:
        return question_seqs, labels, final_samples


def prepare_for_type_2_fewshot(ex, sent_dict, with_definition=False, defs=None, with_prompt=True):
    # sentence_template = "You are given a set of example sentences for a new term or terms and must assess a sentence using it.\nWord: {}\nExamples: {}\nSentence: {}"
    if with_prompt:
        sentence_template = "Here are some sentences for a new word \"{}\":\n{}"
    # definition_template = "You are given a set of example sentences and a definition for a new term or terms and must assess a sentence using it.\nWord: {}\nDefinition: {}.\nExamples: {}\nSentence: {}"

    base_seqs, labels = prepare_for_top_2_selection(ex)
    answers = ex["ANSWERS"][0]
    seqs = []
    question_seqs = []
    final_samples = []
    for w, s in zip(answers, base_seqs):
        nonce = "<nonce>"
        samples = sent_dict[w]
        samples = [re.sub(r"\b({})\b".format(w), nonce, sentence, flags=re.I) for sentence in samples]
        # if with_definition and defs is not None:
        #     definition = defs[w]
        #     def_s = "The word {} is defined as {}".format(nonce, definition)
        #     samples.append(def_s)
        example_string = "\n".join(samples)

        new_s = re.sub(r"\b({})\b".format(w), nonce, s, flags=re.I)
        if with_prompt:
            seq = sentence_template.format(nonce, example_string + "\n" + new_s)
            seqs.append(seq)
        question_seqs.append(new_s)
        final_samples.append(samples)
    if with_prompt:
        return seqs, labels, question_seqs, final_samples
    else:
        return question_seqs, labels, final_samples

def prepare_emb_gen_batch(ex, sent_dict, k, with_def=False, defs=None, with_prompt=False):

    if with_prompt:
        sentence_template = "Here are some sentences for a new word \"{}\":\n{}"

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
            # nonce = "<{}_new>".format(w.lower())
            nonce = "<nonce>"
            samples = sent_dict[w]
            #print(w)
            #print(sent_dict[w])
            # if with_def and defs is not None:
            #     # one less sample because we are using the def as a sample
            #     samples = np.random.choice(
            #         [s for s in sent_dict[w] if re.search(r"\b({})\b".format(w), s, flags=re.I) is not None], size=k-1,
            #         replace=False)
            # else:
            #     samples = np.random.choice([s for s in sent_dict[w] if re.search(r"\b({})\b".format(w), s, flags=re.I) is not None], size=k, replace=False)
            # samples = [s for s in samples if re.search(r"\b({})\b".format(w), s, flags=re.I) is not None]
            samples = [re.sub(r"\b({})\b".format(w), nonce, s, flags=re.I) for s in samples]
            # if with_def and defs is not None:
            #     if w in defs:
            #         definition = defs[w]
            #     else:
            #         definition = defs[w.lower()]
            #     def_s = "The word {} is defined as {}".format(nonce, definition)
            #     samples.append(def_s)
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

    if with_prompt:
        new_task_seqs = []
        for samp, s in zip(task_samples, task_seqs):
            new_task_seq = sentence_template.format("<nonce>","\n".join(samp) +"\n" + s)
            new_task_seqs.append(new_task_seq)

        return task_samples, new_task_seqs, task_seqs, labels
    else:
        return task_samples, task_seqs, labels


@torch.no_grad()
def get_sentence_probs_emb_gen(model, tokenizerMLM, tokenizerTask, contexts, seqs, t5=False):
    probs = []
    for i,seq in enumerate(seqs):
        if t5:
            ctx = [prepare_for_t5(s, "<nonce>") for s in contexts[i]]
            context = tokenizerMLM(ctx, padding=True, truncation=True, max_length=256, return_tensors='pt')
        else:
            context = tokenizerMLM(contexts[i], padding='longest', return_tensors='pt')
        # print(context)
        toks = tokenizerTask(seq, return_tensors="pt").to(model.device)
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
def get_sentence_probs_emb_gen_with_prompt(model, tokenizerMLM, tokenizerTask, contexts, seqs, base_seqs, t5=False):
    probs = []
    for i,(seq, base_seq) in enumerate(zip(seqs, base_seqs)):
        if t5:
            ctx = [prepare_for_t5(s, "<nonce>") for s in contexts[i]]
            context = tokenizerMLM(ctx, padding=True, truncation=True, max_length=256, return_tensors='pt')
        else:
            context = tokenizerMLM(contexts[i], padding='longest', return_tensors='pt')
        # print(context)
        toks = tokenizerTask(seq, return_tensors="pt").to(model.device)
        labels = toks['input_ids'].clone()
        batch = {
            "contexts": [context],
            "input_ids": toks['input_ids'],
            "attention_mask": toks['attention_mask'],
            'labels': labels
        }
        out = model(batch)
        prob = get_sentence_probs_agnostic(out.logits, tokenizerTask, seq, base_seq,
                                           model.secondLM.config.vocab_size + 1)
        probs.append(prob)
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

def evaluate_emb_gen(model, tokenizerMLM, tokenizerTask, ex, sents, k, with_def=False, defs=None, t5=False, with_prompt=False):

    if not with_prompt:
        samples, seqs, labels = prepare_emb_gen_batch(ex, sents, k, with_def, defs, with_prompt=False)
        probs = get_sentence_probs_emb_gen(model, tokenizerMLM, tokenizerTask, samples, seqs, t5=t5)
    else:
        samples, seqs, base_seqs, labels = prepare_emb_gen_batch(ex, sents, k, with_def, defs, with_prompt=True)
        # print(samples)
        # print(seqs)
        # print(base_seqs)
        probs = get_sentence_probs_emb_gen_with_prompt(model, tokenizerMLM, tokenizerTask, samples, seqs, base_seqs, t5=t5)
        # print(probs)
    if ex["ANSWER_TYPE"] == "top_1":
        return evaluate_type_1(probs, labels)
    elif ex["ANSWER_TYPE"] == "top_2":
        return evaluate_type_2(probs, labels)

def prepare_hice_batch(ex, sent_dict, k, with_def=False, defs=None, with_prompt=False):
    if with_prompt:
        sentence_template = "Here are some sentences for a new word \"{}\":\n{}"

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
            # nonce = "<{}_new>".format(w.lower())
            nonce = "<nonce>"
            #print(w)
            print(sent_dict[w])
            samples = sent_dict[w]
            # samples = np.random.choice([s for s in sent_dict[w] if re.search(r"\b({})\b".format(w), s, flags=re.I) is not None], size=k, replace=False)
            samples = [s for s in samples if re.search(r"\b({})\b".format(w), s, flags=re.I) is not None]
            # samples = [re.sub(r"\b({})\b".format(w), nonce, s, flags=re.I) for s in samples]
            # if with_def and defs is not None:
            #     if w in defs:
            #         definition = defs[w]
            #     else:
            #         definition = defs[w.lower()]
            #     def_s = "The word {} is defined as {}".format(nonce, definition)
            #     samples.append(def_s)
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
    if with_prompt:
        new_task_seqs = []
        for samp, s in zip(task_samples, task_seqs):
            new_task_seq = sentence_template.format("\n".join(samp), s)
            new_task_seqs.append(new_task_seq)

        return task_samples, new_task_seqs, task_seqs, labels, answers
    else:
        return task_samples, task_seqs, labels, answers

    # return task_samples, task_seqs, labels, answers

@torch.no_grad()
def get_sentence_probs_hice(model, tokenizerTask, contexts, seqs, answers, dictionary):
    probs = []
    for i,seq in enumerate(seqs):
        b = make_hice_batch(contexts[i], answers[i], dictionary, maxlen=24, pad=0)
        context = b['contexts'].to(model.device)
        vocab = b['character'].to(model.device)
        # print(context.shape)
        # print(vocab.shape)
        toks = tokenizerTask(seq, truncation=True, max_length=256, return_tensors="pt").to(model.device)
        labels = toks['input_ids'].clone()
        batch = {
            "contexts": [context],
            "input_ids": toks['input_ids'],
            "attention_mask": toks['attention_mask'],
            'labels': labels,
            'character': [vocab]
        }
        out = model(batch)
        probs.append(-out.loss.item())
    return probs

@torch.no_grad()
def get_sentence_probs_hice_with_prompt(model, tokenizerTask, contexts, seqs, base_seqs, answers, dictionary):
    probs = []
    for i,(seq, base_seq) in enumerate(zip(seqs, base_seqs)):
        b = make_hice_batch(contexts[i], answers[i], dictionary, maxlen=24, pad=0)
        context = b['contexts'].to(model.device)
        vocab = b['character'].to(model.device)
        # print(context.shape)
        # print(vocab.shape)
        toks = tokenizerTask(seq, truncation=True, max_length=256, return_tensors="pt").to(model.device)
        labels = toks['input_ids'].clone()
        batch = {
            "contexts": [context],
            "input_ids": toks['input_ids'],
            "attention_mask": toks['attention_mask'],
            'labels': labels,
            'character': [vocab]
        }
        out = model(batch)
        prob = get_sentence_probs_agnostic(out.logits, tokenizerTask, seq, base_seq,
                                           model.secondLM.config.vocab_size + 1)
        probs.append(prob)
    return probs


def evaluate_hice(model, tokenizerTask, ex, sents, k, dictionary, with_def=False, defs=None, with_prompt=False):
    if with_prompt:
        samples, seqs, base_seqs, labels, answers = prepare_hice_batch(ex, sents, k, with_def, defs, with_prompt=True)
        probs = get_sentence_probs_hice_with_prompt(model, tokenizerTask, samples, seqs, base_seqs, answers, dictionary)
    else:
        samples, seqs, labels, answers = prepare_hice_batch(ex, sents, k, with_def, defs, with_prompt=False)
        probs = get_sentence_probs_hice(model, tokenizerTask, samples, seqs, answers, dictionary)
    if ex["ANSWER_TYPE"] == "top_1":
        return evaluate_type_1(probs, labels)
    elif ex["ANSWER_TYPE"] == "top_2":
        return evaluate_type_2(probs, labels)


@torch.no_grad()
def get_sentence_probs_additive(model, tokenizerTask, contexts, seqs, answers, dictionary):
    probs = []
    for i,seq in enumerate(seqs):
        b = make_hice_batch(contexts[i], answers[i], dictionary, maxlen=24, pad=0)
        context = b['contexts'].to(model.device)
        # vocab = b['character'].to(model.device)
        # print(context.shape)
        # print(vocab.shape)
        toks = tokenizerTask(seq, truncation=True, max_length=256, return_tensors="pt").to(model.device)
        labels = toks['input_ids'].clone()
        batch = {
            "contexts": [context],
            "input_ids": toks['input_ids'],
            "attention_mask": toks['attention_mask'],
            'labels': labels,
            # 'character': [vocab]
        }
        out = model(batch)
        probs.append(-out.loss.item())
    return probs

@torch.no_grad()
def get_sentence_probs_additive_with_prompt(model, tokenizerTask, contexts, seqs, base_seqs, answers, dictionary):
    probs = []
    for i,(seq,base_seq) in enumerate(zip(seqs, base_seqs)):
        b = make_hice_batch(contexts[i], answers[i], dictionary, maxlen=24, pad=0)
        context = b['contexts'].to(model.device)
        # vocab = b['character'].to(model.device)
        # print(context.shape)
        # print(vocab.shape)
        toks = tokenizerTask(seq, truncation=True, max_length=256, return_tensors="pt").to(model.device)
        labels = toks['input_ids'].clone()
        batch = {
            "contexts": [context],
            "input_ids": toks['input_ids'],
            "attention_mask": toks['attention_mask'],
            'labels': labels,
            # 'character': [vocab]
        }
        out = model(batch)
        prob = get_sentence_probs_agnostic(out.logits, tokenizerTask, seq, base_seq,
                                           model.secondLM.config.vocab_size + 1)
        probs.append(prob)
    return probs



def evaluate_additive(model, tokenizerTask, ex, sents, k, dictionary, with_def=False, defs=None, with_prompt=False):
    if with_prompt:
        samples, seqs, base_seqs, labels, answers = prepare_hice_batch(ex, sents, k, with_def, defs, with_prompt=True)
        probs = get_sentence_probs_additive_with_prompt(model, tokenizerTask, samples, seqs, base_seqs, answers, dictionary)

    else:
        samples, seqs, labels, answers = prepare_hice_batch(ex, sents, k, with_def, defs, with_prompt=False)
        probs = get_sentence_probs_additive(model, tokenizerTask, samples, seqs, answers, dictionary)

    if ex["ANSWER_TYPE"] == "top_1":
        return evaluate_type_1(probs, labels)
    elif ex["ANSWER_TYPE"] == "top_2":
        return evaluate_type_2(probs, labels)
