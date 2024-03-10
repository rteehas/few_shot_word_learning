import json

from tqdm import tqdm

from train_with_llama import *
import torch
import numpy as np
import itertools
import re
from transformers import RobertaForMaskedLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, \
    get_linear_schedule_with_warmup, AdamW, DataCollatorForLanguageModeling, AutoConfig, T5EncoderModel

from torch.nn import CrossEntropyLoss
from w2v_baselines import make_hice_batch, HiCEBaseline, load_dictionary, AdditiveBaseline

device = "cuda"
example_prompt = "The following are examples using a new word <nonce>:\n{}\nThe definition of <nonce> is \"{}\""
def_prompt = "The definition of <nonce> is \"{}\""
base_prompt = " \"{}\""

def prepare_example(ex, k, emb_gen, hice=False, with_prompt=False):
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
    definition = ex['definition'].replace(word, "<nonce>")
    if emb_gen:
        if with_prompt:
            word_seq = example_prompt.format("\n".join(word_samples), definition)
        else:
            word_seq = def_prompt.format(definition)
        base_seq = base_prompt.format(definition)
        seqs.append((word_seq, base_seq))
    else:
        word_seq = def_prompt.format(definition)
        base_seq = base_prompt.format(definition)
        seqs.append((word_seq, base_seq))
    print("word seq", word_seq)
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
        true_words.append(neg)
        neg_samples = np.random.choice([s.replace(neg, "<nonce>").replace(neg.lower(), "<nonce>").replace(neg.capitalize(), "<nonce>").replace(neg.upper(), "<nonce>").replace(" ".join([w.capitalize() for w in neg.split(" ")]), "<nonce>") for s in ex['negative_choice_examples_{}'.format(i)] if neg.lower() in s.lower()], size=k, replace=False).tolist()
        if neg == "beyhive":
            neg_samples = [s.replace("BeyHive", "<nonce>") for s in neg_samples]
        if neg == "goblin mode":
            neg_samples = [s.replace("GOBLIN mode", "<nonce>") for s in neg_samples]
        if emb_gen:
            if with_prompt:
                neg_seq = example_prompt.format("\n".join(neg_samples), definition)
            else:
                neg_seq = def_prompt.format(definition)
            base_seq = base_prompt.format(definition)
            seqs.append((neg_seq, base_seq))
        else:
            neg_seq = example_prompt.format("\n".join(neg_samples), definition)
            base_seq = base_prompt.format(definition)
            seqs.append((neg_seq, base_seq))

        print("neg seq", neg_seq)

        samples.append(neg_samples)
    if hice:
        return samples, seqs, labels, true_words
    else:
        return samples, seqs, labels

# def prepare_example(ex, k, emb_gen, hice=False, with_prompt=False):
#     samples = []
#     seqs = []
#     labels = [1,0,0,0]
#     word = ex['word']
#     true_words = [word]
#     if word == "take the l":
#         word = "take the L"
#     if word == "goblin era":
#         word = "goblin mode"
#     if word == "menty b":
#         word = "menty B"
#     if word == "caught in 4k":
#         word = "caught in 4K"
#     if word == "trade":
#         word = "trad"
#     word_samples = np.random.choice([s.replace(word, "<nonce>").replace(word.lower(), "<nonce>").replace(word.capitalize(), "<nonce>").replace(word.upper(), "<nonce>").replace(" ".join([w.capitalize() for w in word.split(" ")]), "<nonce>") for s in ex['word_examples'] if word.lower() in s.lower()], size=k, replace=False).tolist()
#     if word == "beyhive":
#         word_samples = [s.replace("BeyHive", "<nonce>") for s in word_samples]
#     if word == "goblin mode":
#         word_samples = [s.replace("GOBLIN mode", "<nonce>") for s in word_samples]
#     if word == "l's":
#         word_samples = [s.replace("L\'s", "<nonce>").replace("l\'s", "<nonce>") for s in word_samples]
#
#     definition = ex['definition'].replace(word, "<nonce>")
#     if word == "beyhive":
#         definition = definition.replace("BeyHive", "<nonce>")
#     if word == "goblin mode":
#         definition = definition.replace("GOBLIN mode", "<nonce>")
#     if word == "l's":
#         definition = definition.replace("L\'s", "<nonce>").replace("l\'s", "<nonce>")
#
#     # if emb_gen:
#     if with_prompt:
#         word_seq = example_prompt.format("\n".join(word_samples), definition)
#     else:
#         word_seq = def_prompt.format(definition)
#     base_seq = base_prompt.format(definition)
#     seqs.append((word_seq, base_seq))
#     # else:
#     #     word_seq = example_prompt.format("\n".join(word_samples), definition)
#     #     base_seq = base_prompt.format(definition)
#     #     seqs.append((word_seq, base_seq))
#     print("word seq", word_seq)
#     samples.append(word_samples)
#     # new_ex['word'] = word
#     for i in range(3):
#         neg = ex["negative_choice_{}".format(i)]
#         neg_def = ex["negative_definition_{}".format(i)]
#         if neg == "take the l":
#             neg = "take the L"
#         if neg == "goblin era":
#             neg = "goblin mode"
#         if neg == "menty b":
#             neg = "menty B"
#         if neg == "caught in 4k":
#             neg = "caught in 4K"
#         if neg == "trade":
#             neg = "trad"
#         true_words.append(word)
#         neg_def = re.sub(r"\b({})\b".format(neg), "<nonce>", neg_def, flags=re.I)
#         # neg_samples = np.random.choice([s.replace(neg, "<nonce>").replace(neg.lower(), "<nonce>").replace(neg.capitalize(), "<nonce>").replace(neg.upper(), "<nonce>").replace(" ".join([w.capitalize() for w in neg.split(" ")]), "<nonce>") for s in ex['negative_choice_examples_{}'.format(i)] if neg.lower() in s.lower()], size=k, replace=False).tolist()
#         if neg == "beyhive":
#             neg_def = neg_def.replace("BeyHive", "<nonce>")
#             # neg_samples = [s.replace("BeyHive", "<nonce>") for s in neg_samples]
#         if neg == "goblin mode":
#             neg_def = neg_def.replace("GOBLIN mode", "<nonce>")
#             # neg_samples = [s.replace("GOBLIN mode", "<nonce>") for s in neg_samples]
#         if neg == "l's":
#             neg_def = neg_def.replace("L\'s", "<nonce>").replace("l\'s", "<nonce>")
#         if with_prompt:
#             neg_seq = example_prompt.format("\n".join(word_samples), neg_def)
#         else:
#             neg_seq = def_prompt.format(neg_def)
#         base_seq = base_prompt.format(neg_def)
#         seqs.append((neg_seq, base_seq))
#         # if emb_gen:
#         #     neg_seq = def_prompt.format(definition)
#         #     base_seq = base_prompt.format(definition)
#         #     seqs.append((neg_seq, base_seq))
#         # else:
#         #     neg_seq = example_prompt.format("\n".join(neg_samples), definition)
#         #     base_seq = base_prompt.format(definition)
#         #     seqs.append((neg_seq, base_seq))
#
#         print("neg seq", neg_seq)
#
#         samples.append(word_samples)
#     if hice:
#         return samples, seqs, labels, true_words
#     else:
#         return samples, seqs, labels

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


def evaluate_example_emb_gen(ex, model, tokenizerMLM, tokenizerTask, k, with_prompt):
    samples, seqs, labels = prepare_example(ex, k,emb_gen=True, hice=False, with_prompt=with_prompt)
    probs = []
    for sample, seq_tup in zip(samples, seqs):
        seq, base = seq_tup
        print("sample", sample)
        print("seq", seq)
        print("base", base)

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

def evaluate_example_hice(ex, model, tokenizerTask, k, dictionary, with_prompt):
    samples, seqs, labels, true_words = prepare_example(ex, k, emb_gen=False, hice=True, with_prompt=with_prompt)
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
    samples, seqs, labels, true_words = prepare_example(ex, k, emb_gen=False, hice=True, with_prompt=False)
    probs = []
    for sample, seq_tup, word in zip(samples, seqs, true_words):
        seq, base = seq_tup
        print("sample", sample)
        print("seq", seq)
        b = make_hice_batch(sample,word, dictionary, maxlen=24, pad=0)
        ctx = b['contexts'].to(model.device)
        # vocab = b['character'].to(model.device)
        print(ctx)
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
    # print(probs)
    return evaluate_type_1(probs, labels)
def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--path", type=str)
    parser.add_argument("--setting", type=str, default="emb_gen")
    parser.add_argument("--tuning", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--with_prompt", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser

if __name__ == "__main__":
    device = "cuda"
    twitter_task = load_from_disk("new_twitter_large_v3")
    args = get_arguments().parse_args()
    if args.model == "emb_gen":
        path = args.path
        tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
        tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
        nonces = list(tokenizerTask.get_added_vocab().keys())
        # tokenizerMLM.add_tokens(nonces)
        # tokenizerTask.add_tokens(nonces)
        firstLM = RobertaForMaskedLM.from_pretrained("roberta-large", low_cpu_mem_usage=True)
        # T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        # firstLM = T5EncoderModel.from_pretrained("t5-large", low_cpu_mem_usage=True).to(device)
        secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                    low_cpu_mem_usage=True)
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
        layers = [-1]
        model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), layers, mask_token_id, memory_config, 1, None,
                                      False).to(device)
        model.emb_gen.load_state_dict(torch.load(path + "/pytorch_model.bin"))
        model.device = device
        model.firstLM.eval()
        model.secondLM.eval()

        model.eval()
        scores = {}
        for trial in range(args.trials):

            print("Trial {}".format(trial))
            with torch.no_grad():
                for k in range(1, 5):
                    outputs = []
                    for ex in twitter_task:
                         outputs.append(evaluate_example_emb_gen(ex, model, tokenizerMLM, tokenizerTask, k, args.with_prompt))

                    acc = sum(outputs) / len(outputs)
                    print("Accuracy for k = {} is {}".format(k, acc))
                    if k in scores:
                        scores[k].append(acc)
                    else:
                        scores[k] = [acc]

        fname = "twitter_emb_gen_with_prompt_{}.json".format(args.with_prompt)
        with open(fname, 'w') as fp:
            json.dump(scores, fp)

    elif args.model == "baseline":
        secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                    low_cpu_mem_usage=True).to(device)
        tokenizerTask = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                       legacy=True,
                                                       use_fast=False)
        tokenizerTask.add_tokens(["<nonce>"])
        secondLM.resize_token_embeddings(len(tokenizerTask))
        secondLM.eval()
        scores = {}
        for trial in range(args.trials):

            print("Trial {}".format(trial))
            # with torch.no_grad():
            for k in range(1, 5):
                outputs = []
                for ex in twitter_task:
                     outputs.append(evaluate_example(ex, secondLM, tokenizerTask, k, args.tuning, args.lr))

                # acc = sum(outputs) / len(outputs)
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
        if args.tuning:
            fname = "twitter_baseline_with_prompt_{}_tuning_{}_lr_{}.json".format(args.with_prompt, args.tuning, args.lr)
        else:
            fname = "twitter_baseline_with_prompt_{}_tuning_{}.json".format(args.with_prompt, args.tuning)
        with open(fname, 'w') as fp:
            json.dump(scores, fp)

    elif args.model == "hice":
        hice_path = "HiCE/save/model.pt"
        input_linear_path = "baseline_mappings/input_linear.pt"
        output_linear_path = "baseline_mappings/output_linear.pt"
        w2v_dir = 'HiCE/data/base_w2v/wiki_all.sent.split.model'
        corpus_dir = "HiCE/data/wikitext-103/"

        secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                    low_cpu_mem_usage=True).to(device)
        tokenizerTask = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                       legacy=True,
                                                       use_fast=False)
        tokenizerTask.add_tokens(["<nonce>"])
        secondLM.resize_token_embeddings(len(tokenizerTask))
        secondLM.eval()
        hice = HiCEBaseline(hice_path, input_linear_path, output_linear_path, secondLM).to(device)
        hice.eval()
        hice.device = device
        dictionary = load_dictionary(w2v_dir, corpus_dir, 24)
        scores = {}
        for trial in range(args.trials):

            print("Trial {}".format(trial))
            with torch.no_grad():
                for k in range(1, 5):
                    outputs = []
                    for ex in twitter_task:
                         outputs.append(evaluate_example_hice(ex, hice, tokenizerTask, k, dictionary, with_prompt=args.with_prompt))

                    acc = sum(outputs) / len(outputs)
                    print("Accuracy for k = {} is {}".format(k, acc))
                    if k in scores:
                        scores[k].append(acc)
                    else:
                        scores[k] = [acc]
        # if args.tuning:
        fname = "twitter_hice_with_prompt_{}_.json".format(args.with_prompt)
        with open(fname, 'w') as fp:
            json.dump(scores, fp)
    elif args.model == "additive":
        secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                    low_cpu_mem_usage=True).to(device)

        tokenizerTask = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                       legacy=True,
                                                       use_fast=False)
        tokenizerTask.pad_token = tokenizerTask.unk_token
        tokenizerTask.add_tokens(["<nonce>"])

        hice_path = "HiCE/save/model.pt"
        input_linear_path = "baseline_mappings/input_linear.pt"
        output_linear_path = "baseline_mappings/output_linear.pt"
        w2v_dir = 'HiCE/data/base_w2v/wiki_all.sent.split.model'
        corpus_dir = "HiCE/data/wikitext-103/"
        dictionary = load_dictionary(w2v_dir, corpus_dir, 24)
        additive = AdditiveBaseline(dictionary, input_linear_path, output_linear_path, secondLM).to(device)
        additive.device = device
        scores = {}
        for trial in range(args.trials):

            print("Trial {}".format(trial))
            with torch.no_grad():
                for k in range(1, 5):
                    outputs = []
                    for ex in twitter_task:
                         outputs.append(evaluate_example_additive(ex, additive, tokenizerTask, k, dictionary))

                    acc = sum(outputs) / len(outputs)
                    print("Accuracy for k = {} is {}".format(k, acc))
                    if k in scores:
                        scores[k].append(acc)
                    else:
                        scores[k] = [acc]
        # if args.tuning:
        fname = "twitter_additive_with_prompt_{}_.json".format(args.with_prompt)
        with open(fname, 'w') as fp:
            json.dump(scores, fp)

    # elif args.model == "additive":



