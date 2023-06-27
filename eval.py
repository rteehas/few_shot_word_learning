import math
from argparse import ArgumentParser
from functools import reduce
import os
from copy import deepcopy
from scipy import stats
import json
import higher
import csv
from accelerate import Accelerator

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaForMaskedLM, GPT2Model, RobertaForQuestionAnswering, \
    AutoModelForSequenceClassification, AutoModelForCausalLM
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_from_disk

from configs.config import *
from eval_utils import compute_exact_match, load_model_partial
from modules.buffer import RetrievalBuffer
from modules.model import MorphMemoryModel, MorphMemoryModelSQuAD, MorphMemoryModelSNLI, MorphMemoryModelGPT, \
    MorphMemoryModelGPTOnline, MorphMemoryModelGPTSubtoken
from data.data_utils import *
from train_utils import *
from data.few_shot_datasets import *


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-6)
    # parser.add_argument("--warmup", type=int, default=1e2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--model_dir", type=str, default="../few_shot_word_learning/checkpoints/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--taskName", type=str, default='mlm')
    parser.add_argument("--memory", type=str, default="mean")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--intermediate_loss", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--maml", action="store_true")
    parser.add_argument("--word_path", type=str, default='')
    parser.add_argument("--finetune", action="store_true")
    return parser


if __name__ == "__main__":
    args = get_arguments().parse_args()

    print("Arguments: ", args)
    accelerator = Accelerator()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = accelerator.device
    layers = [-1] # add arg to pass this

    tokenizerMLM = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    firstLM = RobertaForMaskedLM.from_pretrained('roberta-base')

# change these to accept model names
    if args.taskName == "mlm":
        tokenizerTask = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)
        secondLM = RobertaForMaskedLM.from_pretrained('roberta-base')

    elif args.taskName == "autoregressive":
        tokenizerTask = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
        secondLM = GPT2Model.from_pretrained("gpt2")

    elif args.taskName == "squad":
        tokenizerTask = AutoTokenizer.from_pretrained('deepset/tinyroberta-squad2', use_fast=True)
        secondLM = RobertaForQuestionAnswering.from_pretrained("deepset/tinyroberta-squad2")

    elif args.taskName == "snli":
        tokenizerTask = AutoTokenizer.from_pretrained('cross-encoder/nli-roberta-base', use_fast=True)
        secondLM = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-roberta-base').to(device)
    elif args.taskName == "addition":
        tokenizerTask = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
        tokenizerTask.pad_token = tokenizerTask.unk_token
        secondLM = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        nonces = ["<OP>"]
        dataset_name = "addition"
    elif args.taskName == "addition_subtok":
        tokenizerTask = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
        tokenizerTask.pad_token = tokenizerTask.unk_token
        secondLM = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        nonces = ["<NUM1>", "<NUM2>"]
        dataset_name = "addition_subtok"

    elif args.taskName == "online":
        tokenizerTask = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
        tokenizerTask.pad_token = tokenizerTask.unk_token
        secondLM = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        with open(args.word_path, 'r') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        words = rows[0]
        nonces = list(map(lambda w: "<{}_new>".format(w), words))
    else:
        raise NotImplementedError("{} not implemented".format(args.taskName))

    if "addition" not in args.taskName:
        dataset = load_from_disk(args.data_path)
        if args.taskName == "online":
            dataset = dataset.filter(lambda ex: len(ex['text']) > 10)


    if "chimera" in args.data_path:
        nonces = list(map(make_nonce, list(set(dataset["train"]['id'] + dataset["test"]['id']))))
        dataset_name = "chimera{}".format(args.num_examples)

    elif "sanity" in args.data_path:
        nonces = list(map(make_sanity_nonce, list(set(dataset['word']))))
        dataset_name = "sanity"

    elif "squad" in args.data_path:
        dataset = dataset.filter(lambda ex: ex['replace'] != '')
        nonces = list(set(map(lambda e: "<{}_nonce>".format(e.lower()), dataset['replace'])))
        dataset_name = "squad"
    elif "snli" in args.data_path:
        nonces = list(map(snli_nonce, list(set(dataset['replace']))))
        dataset_name = "snli"
    elif "wikitext" in args.data_path:
        dataset_name = "online"
    else:
        if "addition" not in args.taskName:
            raise NotImplementedError("Not implemented for this dataset")

    tokenizerMLM.add_tokens(nonces)
    tokenizerTask.add_tokens(nonces)

    # resize models
    firstLM.resize_token_embeddings(len(tokenizerMLM))
    secondLM.resize_token_embeddings(len(tokenizerTask))

    # memory
    if "mean" in args.checkpoint:
        memory_config = AggregatorConfig()
        weight_decay = 0.01

    elif "rnn" in args.checkpoint:
        memory_config = RNNAggConfig()
        weight_decay = 0.015
    elif "cls" in args.checkpoint:
        memory_config = TransformerCLSConfig()
        weight_decay = 0.015
    else:
        raise NotImplementedError("This memory aggregation is not implemented")

    mask_token_id = tokenizerMLM.mask_token_id

    if "snli" in args.data_path:
        n = args.num_examples
        split = dataset.train_test_split(0.2)
        train = SimpleSNLIDataset(split["train"], tokenizerMLM, tokenizerTask, n)
        test = SimpleSNLIDataset(split["test"], tokenizerMLM, tokenizerTask, n)

        train_dl = DataLoader(train, batch_size=args.batch_size, collate_fn=make_collate(train), shuffle=True)
        test_dl = DataLoader(test, batch_size=args.batch_size, collate_fn=make_collate(test))
    else:
        raise NotImplementedError
    # else:
    #     if args.taskName == "calc":

    new_toks = tokenizerMLM.convert_tokens_to_ids(nonces)
    epochs = args.epochs
    lr = args.lr
    epsilon = 1e-8

    if "snli" in args.data_path:
        test_model = MorphMemoryModelSNLI(firstLM, secondLM, new_toks, device, [-1],
                                   tokenizerMLM.mask_token_id, memory_config, "Transformer").to(device)
        test_model = load_model_partial(args.model_dir + args.checkpoint, test_model)

    else:
        raise NotImplementedError

    opt = AdamW(filter(lambda p: p.requires_grad, test_model.parameters()),
                lr=lr,
                eps=epsilon,
                weight_decay=weight_decay
                )

    warmup_steps = 3e2
    eval_ind = len(train_dl) // 2
    if args.taskName == "addition":
        eval_ind = 30

    scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, epochs * len(train_dl))
    intermediate = args.intermediate_loss

    test_model.to(device)
    test_model, opt, train_dl, test_dl, scheduler = accelerator.prepare(
        test_model, opt, train_dl, test_dl, scheduler
    )

    project = "fewshot_model_testing_finetune"
    run = wandb.init(project=project, reinit=True)
    wandb.run.name = "finetuned_{}_{}examples_{}_{}_bs={}_6layers".format(dataset_name,
                                                                    args.num_examples, lr,
                                                                    memory_config.agg_method, args.batch_size)
    eval_ind = 300

    best_corr = 0
    best_acc = 0
    best_loss = 10000
    n_inner_iter = 3
    for epoch in range(epochs):
        train_corr = []
        train_losses = []
        train_correct = 0
        train_total = 0
        for i, batch in enumerate(train_dl):
            log_dict = {}

            test_model.train()
            if not args.maml:
                test_model.zero_grad()
                opt.zero_grad()


                out, losses = test_model(batch)

                loss = out.loss

                log_dict['train loss'] = loss.item()
                train_losses.append(loss.item())

            if "snli" in args.data_path:
                preds = out.logits
                preds = F.log_softmax(preds, dim=-1).argmax(dim=1)
                true_ans = batch['task_labels'].to(device).view(-1)
                num_correct = (preds == true_ans).sum()
                train_correct += num_correct
                train_total += batch['task_labels'].shape[0]

            if intermediate:
                intermediate_loss = torch.sum(torch.Tensor([losses]))
                log_dict["emb generator loss"] = intermediate_loss.item()
                final_loss = loss + intermediate_loss
            else:
                final_loss = loss

            # final_loss.backward()
            accelerator.backward(final_loss)
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, test_model.parameters()), 1.0)
            for name, param in test_model.emb_gen.named_parameters():
                log_dict["post_{}_grad_norm".format(name)] = torch.norm(param.grad.view(-1)).item()
                if torch.isnan(torch.norm(param.grad.view(-1))):
                    raise Exception("Nan Gradient")

            opt.step()
            scheduler.step()
            with torch.no_grad():
                for v in test_model.memory.memory:
                    for val in test_model.memory.memory[v]:
                        del val
            test_model.memory.memory = {}
            wandb.log(log_dict)

            if i != 0 and (i % eval_ind == 0 or i % len(train_dl) == 0):
                opt.zero_grad(set_to_none=True)
                test_model.eval()
                if "snli" in args.data_path:
                    test_model.eval()
                    total_correct = 0
                    total = 0
                    for b in test_dl:
                        t_out, _ = test_model.forward(b)
                        preds = t_out.logits
                        preds = F.log_softmax(preds, dim=-1).argmax(dim=1)
                        true_ans = b['task_labels'].to(device).view(-1)
                        num_correct = (preds == true_ans).sum()
                        total_correct += num_correct
                        total += b['task_labels'].shape[0]
                        with torch.no_grad():
                            for v in test_model.memory.memory:
                                for val in test_model.memory.memory[v]:
                                    del val
                        test_model.memory.memory = {}
                    acc = total_correct / total
                    wandb.log({'epoch': epoch, 'average test accuracy': acc})
                    if best_acc < acc:
                        chkpt_name = get_model_name_checkpoint(wandb.run.name, epoch)
                        save(test_model, opt, chkpt_name)
                        print("Saved {}".format(chkpt_name))
                        best_acc = acc

        wandb.log({"epoch": epoch, 'average train loss': sum(train_losses) / len(train_losses)})
        if "snli" in args.data_path:
            wandb.log({"epoch": epoch, 'average train acc': train_correct / train_total})
