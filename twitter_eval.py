from train_with_llama import *
import torch
import numpy as np
import itertools
import re

from torch.nn import CrossEntropyLoss
from w2v_baselines import make_hice_batch
device = "cuda"
example_prompt = "The following are examples using a new word <nonce>:\n{}\nThe definition of <nonce> is \"{}\""
def_prompt = "The definition of <nonce> is \"{}\""
base_prompt = " \"{}\""

def prepare_example(ex, k, hice=False, with_prompt=False):
    samples = []
    seqs = []
    labels = [1,0,0,0]
    word = ex['word']
    true_words = [word]
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
    if word == "l's":
        word_samples = [s.replace("L\'s", "<nonce>").replace("l\'s", "<nonce>") for s in word_samples]

    definition = ex['definition'].replace(word, "<nonce>")
    if word == "beyhive":
        definition = definition.replace("BeyHive", "<nonce>")
    if word == "goblin mode":
        definition = definition.replace("GOBLIN mode", "<nonce>")
    if word == "l's":
        definition = definition.replace("L\'s", "<nonce>").replace("l\'s", "<nonce>")

    # if emb_gen:
    if with_prompt:
        word_seq = example_prompt.format("\n".join(word_samples), definition)
    else:
        word_seq = def_prompt.format(definition)
    base_seq = base_prompt.format(definition)
    seqs.append((word_seq, base_seq))
    # else:
    #     word_seq = example_prompt.format("\n".join(word_samples), definition)
    #     base_seq = base_prompt.format(definition)
    #     seqs.append((word_seq, base_seq))
    print("word seq", word_seq)
    samples.append(word_samples)
    # new_ex['word'] = word
    for i in range(3):
        neg = ex["negative_choice_{}".format(i)]
        neg_def = ex["negative_definition_{}".format(i)]
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
        true_words.append(word)
        neg_def = re.sub(r"\b({})\b".format(neg), "<nonce>", neg_def, flags=re.I)
        # neg_samples = np.random.choice([s.replace(neg, "<nonce>").replace(neg.lower(), "<nonce>").replace(neg.capitalize(), "<nonce>").replace(neg.upper(), "<nonce>").replace(" ".join([w.capitalize() for w in neg.split(" ")]), "<nonce>") for s in ex['negative_choice_examples_{}'.format(i)] if neg.lower() in s.lower()], size=k, replace=False).tolist()
        if neg == "beyhive":
            neg_def = neg_def.replace("BeyHive", "<nonce>")
            # neg_samples = [s.replace("BeyHive", "<nonce>") for s in neg_samples]
        if neg == "goblin mode":
            neg_def = neg_def.replace("GOBLIN mode", "<nonce>")
            # neg_samples = [s.replace("GOBLIN mode", "<nonce>") for s in neg_samples]
        if neg == "l's":
            neg_def = neg_def.replace("L\'s", "<nonce>").replace("l\'s", "<nonce>")
        if with_prompt:
            neg_seq = example_prompt.format("\n".join(word_samples), neg_def)
        else:
            neg_seq = def_prompt.format(neg_def)
        base_seq = base_prompt.format(neg_def)
        seqs.append((neg_seq, base_seq))
        # if emb_gen:
        #     neg_seq = def_prompt.format(definition)
        #     base_seq = base_prompt.format(definition)
        #     seqs.append((neg_seq, base_seq))
        # else:
        #     neg_seq = example_prompt.format("\n".join(neg_samples), definition)
        #     base_seq = base_prompt.format(definition)
        #     seqs.append((neg_seq, base_seq))

        print("neg seq", neg_seq)

        samples.append(word_samples)
    if hice:
        return samples, seqs, labels, true_words
    else:
        return samples, seqs, labels

def prepare_prompt(examples, definition):
    return example_prompt.format("\n".join(examples), definition)

def evaluate_example(ex, model, tokenizer, k, tuning=False, lr=3e-4):
    samples, seqs, labels = prepare_example(ex, k, False)


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
        for sample, seq_tup in zip(samples, seqs):
            opt = AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=lr)
            input = tokenizer(sample, truncation=True, padding='longest', return_tensors='pt')
            per_step_probs = []
            seq, base_seq = seq_tup
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
                    prob = get_sentence_probs_agnostic(logits, tokenizer, seq, base_seq, model.config.vocab_size)
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
            for seq_tup in seqs:
                seq, base_seq = seq_tup
                inputs = tokenizer(seq, truncation=True, return_tensors='pt').to(device)
                out = model(**inputs)
                logits = out.logits
                prob = get_sentence_probs_agnostic(logits, tokenizer, seq, base_seq, model.config.vocab_size)
                probs.append(prob)
            print()
            print("probs",probs)
            return evaluate_type_1(probs, labels)

def evaluate_type_1(probs, labels):
    probs = np.array(probs)
    max_prob = np.argmax(probs)
    lab_id = labels.index(1)

    return max_prob == lab_id

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
        print("loss",loss)
        probs.append(-loss.item())
    return probs

def get_sentence_probs_agnostic(logits, tokenizer, seq, base, vocab_size):
    ce = CrossEntropyLoss()
    toks = tokenizer(seq, return_tensors="pt").to(device)
    question_toks = tokenizer(base)
    # print(logits.shape, toks['input_ids'].shape,len(question_toks['input_ids']) )
    answer_length = len(question_toks['input_ids']) - 1
    labels = toks['input_ids'].clone()
    if len(logits.shape) == 2:
        logits = logits.unsqueeze(0)
    answer_labels = labels[:, -answer_length:]
    answer_logits = logits[:, -answer_length:, :]
    shift_logits = answer_logits[..., :-1, :].contiguous()
    shift_labels = answer_labels[..., 1:].contiguous()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    loss = ce(shift_logits, shift_labels)
    return -loss.item()


# def get_sentence_prob(labels, logits, model):
#     ce = CrossEntropyLoss(reduction='none')
#     # print(answer_labels)
#     shift_logits = logits[..., :-1, :].contiguous()
#     shift_labels = labels[..., 1:].contiguous()
#     shift_logits = shift_logits.view(-1, model.config.vocab_size + 1)
#     shift_labels = shift_labels.view(-1)
#     shift_labels = shift_labels.to(shift_logits.device)
#     loss = ce(shift_logits, shift_labels)
#     return -loss[-1].item()


def evaluate_example_emb_gen(ex, model, tokenizerMLM, tokenizerTask, k):
    samples, seqs, labels = prepare_example(ex, k, True)
    probs = []
    for sample, seq_tup in zip(samples, seqs):
        seq, base = seq_tup
        print("sample", sample)
        print("seq", seq)

        ctx = tokenizerMLM(sample, truncation=True, padding='longest', return_tensors='pt').to(device)
        input = tokenizerTask(seq,truncation=True, return_tensors='pt').to(device)
        batch = {
            'contexts': [ctx],
            'input_ids': input['input_ids'],
            'attention_mask': input['attention_mask'],
            'labels': input['input_ids'].clone()
        }

        outputs = model(batch)
        # prob = get_sentence_prob(input['input_ids'].clone(), outputs.logits, model.secondLM)
        prob = get_sentence_probs_agnostic(outputs.logits, tokenizerTask, seq, base, model.secondLM.config.vocab_size + 1)
        # prob = -outputs.loss.item()
        probs.append(prob)
    print(probs)
    return evaluate_type_1(probs, labels)

def get_def_loss_emb_gen(ex, model, tokenizerMLM, tokenizerTask, k):
    samples, seqs, labels = prepare_example(ex, k, True)
    sample = samples[0]
    seq = seqs[0]
    ctx = tokenizerMLM(sample, truncation=True, padding='longest', return_tensors='pt').to(device)
    input = tokenizerTask(seq, truncation=True, return_tensors='pt').to(device)
    batch = {
        'contexts': [ctx],
        'input_ids': input['input_ids'],
        'attention_mask': input['attention_mask'],
        'labels': input['input_ids'].clone()
    }

    outputs = model(batch)
    return outputs.loss.item()

def get_def_loss_baseline(ex, model, tokenizer, k, tuning=False, lr=3e-4):
    samples, seqs, labels = prepare_example(ex, k, False)
    sample = samples[0]
    seq, base_seq = seqs[0]

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
                # inputs = tokenizer(seq, truncation=True, return_tensors='pt').to(device)
                # out = model(**inputs)
                # logits = out.logits
                prob = get_sentence_probs(model, tokenizer, [seq], [base_seq])
                per_step_probs.append(-prob[0])


        model.get_input_embeddings().weight = torch.nn.Parameter(orig_input_embeds)
        model.get_output_embeddings().weight = torch.nn.Parameter(orig_output_embeds)

        return per_step_probs

    else:
        with torch.no_grad():
            model.eval()
                # inputs = tokenizer(seq, truncation=True, return_tensors='pt').to(device)
                # out = model(**inputs)
                # logits = out.logits
            prob = get_sentence_probs(model, tokenizer, [seq], [base_seq])
            return -prob[0]

def evaluate_example_hice(ex, model, tokenizerTask, k, dictionary):
    samples, seqs, labels, true_words = prepare_example(ex, k, False, hice=True)
    probs = []
    for sample, seq_tup, word in zip(samples, seqs, true_words):
        seq, base = seq_tup
        print("sample", sample)
        print("seq", seq)
        b = make_hice_batch(sample,word, dictionary, maxlen=24, pad=0)
        ctx = b['contexts'].to(model.device)
        vocab = b['character'].to(model.device)

        # ctx = tokenizerMLM(sample, truncation=True, padding='longest', return_tensors='pt').to(device)
        input = tokenizerTask(seq, truncation=True, return_tensors='pt').to(device)
        batch = {
            'contexts': [ctx],
            'input_ids': input['input_ids'],
            'attention_mask': input['attention_mask'],
            'labels': input['input_ids'].clone(),
            'character': [vocab]
        }

        outputs = model(batch)
        # prob = get_sentence_prob(input['input_ids'].clone(), outputs.logits, model.secondLM)
        prob = get_sentence_probs_agnostic(outputs.logits, tokenizerTask, seq, base, model.secondLM.config.vocab_size + 1)
        # prob = -outputs.loss.item()
        probs.append(prob)
    print(probs)
    return evaluate_type_1(probs, labels)

def evaluate_example_additive(ex, model, tokenizerTask, k, dictionary):
    samples, seqs, labels, true_words = prepare_example(ex, k, False, hice=True)
    probs = []
    for sample, seq_tup, word in zip(samples, seqs, true_words):
        seq, base = seq_tup
        print("sample", sample)
        print("seq", seq)
        b = make_hice_batch(sample,word, dictionary, maxlen=24, pad=0)
        ctx = b['contexts'].to(model.device)
        # vocab = b['character'].to(model.device)

        # ctx = tokenizerMLM(sample, truncation=True, padding='longest', return_tensors='pt').to(device)
        input = tokenizerTask(seq, truncation=True, return_tensors='pt').to(device)
        batch = {
            'contexts': [ctx],
            'input_ids': input['input_ids'],
            'attention_mask': input['attention_mask'],
            'labels': input['input_ids'].clone(),
            # 'character': [vocab]
        }

        outputs = model(batch)
        # prob = get_sentence_prob(input['input_ids'].clone(), outputs.logits, model.secondLM)
        prob = get_sentence_probs_agnostic(outputs.logits, tokenizerTask, seq, base, model.secondLM.config.vocab_size + 1)
        # prob = -outputs.loss.item()
        probs.append(prob)
    print(probs)
    return evaluate_type_1(probs, labels)
