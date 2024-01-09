from train_with_llama import *
import torch
import numpy as np
import itertools
import re

from torch.nn import CrossEntropyLoss
device = "cuda"
example_prompt = "The following are examples using a new word <nonce>:\n{}\nThe definition of this word is \"{}\". The word is <nonce>"
def_prompt = "The definition of a new word is \"{}\". The word is <nonce>"

def prepare_example(ex, k, emb_gen):
    samples = []
    seqs = []
    labels = [1,0,0,0]
    word = ex['word']
    if word == "take the l":
        word = "take the L"
    if word == "goblin era":
        word = "goblin mode"
    if word == "menty b":
        word = "menty B"
    if word == "caught in 4k":
        word = "caught in 4K"
    if word == "trade":
        word = "trad"
    word_samples = np.random.choice([s.replace(word, "<nonce>").replace(word.lower(), "<nonce>").replace(word.capitalize(), "<nonce>").replace(word.upper(), "<nonce>").replace(" ".join([w.capitalize() for w in word.split(" ")]), "<nonce>") for s in ex['word_examples'] if word.lower() in s.lower()], size=k, replace=False).tolist()
    if word == "beyhive":
        word_samples = [s.replace("BeyHive", "<nonce>") for s in word_samples]
    if word == "goblin mode":
        word_samples = [s.replace("GOBLIN mode", "<nonce>") for s in word_samples]
    definition = ex['definition'].replace(word, "<nonce>")
    if emb_gen:
        word_seq = def_prompt.format(definition)
    else:
        word_seq = example_prompt.format("\n".join(word_samples), definition)
    print("word seq", word_seq)
    seqs.append(word_seq)
    samples.append(word_samples)
    # new_ex['word'] = word
    for i in range(3):
        neg = ex["negative_choice_{}".format(i)]
        if neg == "take the l":
            neg = "take the L"
        if neg == "goblin era":
            neg = "goblin mode"
        if neg == "menty b":
            neg = "menty B"
        if neg == "caught in 4k":
            neg = "caught in 4K"
        if neg == "trade":
            neg = "trad"

        neg_samples = np.random.choice([s.replace(neg, "<nonce>").replace(neg.lower(), "<nonce>").replace(neg.capitalize(), "<nonce>").replace(neg.upper(), "<nonce>").replace(" ".join([w.capitalize() for w in neg.split(" ")]), "<nonce>") for s in ex['negative_choice_examples_{}'.format(i)] if neg.lower() in s.lower()], size=k, replace=False).tolist()
        if neg == "beyhive":
            neg_samples = [s.replace("BeyHive", "<nonce>") for s in neg_samples]
        if neg == "goblin mode":
            neg_samples = [s.replace("GOBLIN mode", "<nonce>") for s in neg_samples]
        if emb_gen:
            neg_seq = def_prompt.format(definition)
        else:
            neg_seq = example_prompt.format("\n".join(neg_samples), definition)
        print("neg seq", neg_seq)
        seqs.append(neg_seq)
        samples.append(neg_samples)
    return samples, seqs, labels

def prepare_prompt(examples, definition):
    return example_prompt.format("\n".join(examples), definition)

def evaluate_example(ex, model, tokenizer, k, tuning=False, lr=3e-4):
    samples, seqs, labels = prepare_example(ex, k)


    tokenizer.pad_token = tokenizer.unk_token
    orig_input_embeds = model.get_input_embeddings().weight.clone()
    orig_output_embeds = model.get_output_embeddings().weight.clone()

    if tuning:
        max_steps = 2
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
        for sample, seq in zip(samples, seqs):
            opt = AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=lr)
            input = tokenizer(sample, truncation=True, padding='longest', return_tensors='pt')
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
                    inputs = tokenizer(seq, truncation=True, return_tensors='pt').to(device)
                    out = model(**inputs)
                    logits = out.logits
                    prob = get_sentence_prob(inputs['input_ids'].clone(), logits, model)
                    per_step_probs.append(prob)

            total_probs.append(per_step_probs)
            model.get_input_embeddings().weight = torch.nn.Parameter(orig_input_embeds)
            model.get_output_embeddings().weight = torch.nn.Parameter(orig_output_embeds)

        seq_probs_by_step = [[seq_prob[step] for seq_prob in total_probs] for step in range(max_steps)]
        example_outputs_by_step = []
        for prob in seq_probs_by_step:
            example_outputs_by_step.append(evaluate_type_1(prob, labels))
        return example_outputs_by_step

    else:
        with torch.no_grad():
            model.eval()
            probs = []
            for seq in seqs:
                inputs = tokenizer(seq, truncation=True, return_tensors='pt').to(device)
                out = model(**inputs)
                logits = out.logits
                prob = get_sentence_prob(inputs['input_ids'].clone(), logits, model)
                probs.append(prob)

            return evaluate_type_1(probs, labels)

def evaluate_type_1(probs, labels):
    probs = np.array(probs)
    max_prob = np.argmax(probs)
    lab_id = labels.index(1)

    return max_prob == lab_id

def get_sentence_prob(labels, logits, model):
    ce = CrossEntropyLoss(reduction='none')
    # print(answer_labels)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_logits = shift_logits.view(-1, model.config.vocab_size + 1)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    loss = ce(shift_logits, shift_labels)
    return -loss[-1].item()


def evaluate_example_emb_gen(ex, model, tokenizerMLM, tokenizerTask, k):
    samples, seqs, labels = prepare_example(ex, k)
    probs = []
    for sample, seq in zip(samples, seqs):
        print(sample)
        ctx = tokenizerMLM(sample, truncation=True, padding='longest', return_tensors='pt').to(device)
        input = tokenizerTask(seq,truncation=True, return_tensors='pt').to(device)
        batch = {
            'contexts': [ctx],
            'input_ids': input['input_ids'],
            'attention_mask': input['attention_mask'],
            'labels': input['input_ids'].clone()
        }

        outputs = model(batch)
        prob = get_sentence_prob(input['input_ids'].clone(), outputs.logits, model.secondLM)
        probs.append(prob)
    return evaluate_type_1(probs, labels)

