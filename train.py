import math
from argparse import ArgumentParser
from functools import reduce
import os
from copy import deepcopy
from scipy import stats
import json

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaForMaskedLM, GPT2Model, RobertaForQuestionAnswering, \
    AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_from_disk

from configs.config import *
from modules.model import MorphMemoryModel, MorphMemoryModelSQuAD, MorphMemoryModelSNLI
from data.data_utils import *
from train_utils import *
from data.few_shot_datasets import *

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-6)
    # parser.add_argument("--warmup", type=int, default=1e2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--tgt_data_path", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load_checkpoint", type=bool, default=False)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--taskName", type=str, default='mlm')
    parser.add_argument("--memory", type=str, default="mean")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--intermediate_loss", type=bool, default=False)
    parser.add_argument("--trial", type=str, default='l2')
    parser.add_argument("--emb_gen", type=str, default='mlp')
    parser.add_argument("--strategy", type=str, default='mask')
    return parser


if __name__ == "__main__":
    args = get_arguments().parse_args()

    print("Arguments: ", args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    else:
        raise NotImplementedError("{} not implemented".format(args.taskName))

    dataset = load_from_disk(args.data_path)

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

    else:
        raise NotImplementedError("Not implemented for this dataset")

    # add support for other datasets

    # expand tokenizer
    tokenizerMLM.add_tokens(nonces)
    tokenizerTask.add_tokens(nonces)

    # resize models
    firstLM.resize_token_embeddings(len(tokenizerMLM))
    secondLM.resize_token_embeddings(len(tokenizerTask))

    # memory
    if args.memory == "mean":
        memory_config = AggregatorConfig()

    elif args.memory == "rnn":
        memory_config = RNNAggConfig()
    elif args.memory == "cls":
        memory_config = TransformerCLSConfig()
    else:
        raise NotImplementedError("This memory aggregation is not implemented")

    mask_token_id = tokenizerMLM.mask_token_id

    if "chimera" in args.data_path:

        cos = nn.CosineSimilarity(dim=-1)

        split = dataset["train"].train_test_split(test_size=0.2)
        mlm_dataset = ChimeraMLMDataset(split["train"], tokenizerMLM, tokenizerTask, args.num_examples, args.trial)

        train_dl = DataLoader(mlm_dataset, batch_size=1, shuffle=True)

        test_dataset = ChimeraTestDataset(split["test"], tokenizerMLM, tokenizerTask, args.num_examples, args.trial)

        collate = make_collate(test_dataset)

        test_dl = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate)

    if "sanity" in args.data_path:
        split = dataset.train_test_split(test_size=0.2)
        n = args.num_examples
        tokenizerMLM.add_tokens(nonces)
        tokenizerTask.add_tokens(nonces)

        mlm_dataset = SimpleMLMDataset(split["train"], tokenizerMLM, tokenizerTask, n)

        train_dl = DataLoader(mlm_dataset, batch_size=1, shuffle=True)

        # train_eval = ChimeraTestDataset(chimera["train"], tokenizerMLM, tokenizerTask, n, trial)

        test_dataset = SimpleMLMDataset(split["test"], tokenizerMLM, tokenizerTask, n)

        # collate = make_collate(test_dataset)

        test_dl = DataLoader(test_dataset, batch_size=1, shuffle=True)

    if "squad" in args.data_path:

        tokenizerMLM.add_tokens(nonces)
        tokenizerTask.add_tokens(nonces)

        split = dataset.train_test_split(0.2)

        train = SimpleSQuADDataset(split['train'], tokenizerMLM, tokenizerTask, args.num_examples)

        test = SimpleSQuADDataset(split['test'], tokenizerMLM, tokenizerTask, args.num_examples)

        train_dl = DataLoader(train, batch_size=5, collate_fn=make_collate(train), shuffle=True)

        test_dl = DataLoader(test, batch_size=5, collate_fn=make_collate(test))

    if "snli" in args.data_path:
        n=args.num_examples
        split = dataset.train_test_split(0.2)
        train = SimpleSNLIDataset(split["train"], tokenizerMLM, tokenizerTask, n)
        test = SimpleSNLIDataset(split["test"], tokenizerMLM, tokenizerTask, n)

        train_dl = DataLoader(train, batch_size=5, collate_fn=make_collate(train), shuffle=True)
        test_dl = DataLoader(test, batch_size=5, collate_fn=make_collate(test))

    new_toks = tokenizerTask.convert_tokens_to_ids(nonces)


    epochs = args.epochs
    lr = args.lr
    epsilon = 1e-8

    if "squad" in args.data_path:
        test_model = MorphMemoryModelSQuAD(firstLM, secondLM, new_toks,
                                      device, layers, mask_token_id, memory_config, args.emb_gen).to(device)
    elif "snli" in args.data_path:
        test_model = MorphMemoryModelSNLI(firstLM, secondLM, new_toks, device, [-1],
                                   tokenizerMLM.mask_token_id, memory_config, args.emb_gen).to(device)
    elif "sanity" in args.data_path:
        test_model = MorphMemoryModel(firstLM, secondLM, new_toks,
                                  device, layers, mask_token_id, memory_config, args.emb_gen).to(device)
    else:
        raise NotImplementedError

    opt = AdamW(filter(lambda p: p.requires_grad, test_model.parameters()),
                lr=lr,
                eps=epsilon
                )

    warmup_steps = 3e2
    eval_ind = len(train_dl) // 2
    scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, epochs * len(train_dl))
    intermediate = args.intermediate_loss

    run = wandb.init(project="fewshot_model_testing_redone", reinit=True)
    wandb.run.name = "{}_{}_{}_{}".format(dataset_name, lr, memory_config.agg_method, args.emb_gen)

    if intermediate:
        wandb.run.name = wandb.run.name + "_intermediate"



    best_corr = 0
    best_acc = 0
    best_loss = 10000
    for epoch in range(epochs):
        train_corr = []
        train_losses = []
        train_correct = 0
        train_total = 0
        for i, batch in enumerate(train_dl):
            log_dict = {}

            test_model.train()

            test_model.zero_grad()
            opt.zero_grad()
            out, losses = test_model(batch)

            loss = out.loss

            log_dict['train loss'] = loss.item()
            train_losses.append(loss.item())
            if "sanity" in args.data_path:
                nonce_loss = get_nonce_loss(batch, out, test_model.secondLM.vocab_size, device)
                if nonce_loss:
                    log_dict["new token loss"] = nonce_loss.item()
            elif "snli" in args.data_path:
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

            final_loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, test_model.parameters()), 1.0)
            opt.step()
            scheduler.step()
            test_model.memory.memory = {}
            wandb.log(log_dict)

            if (i + 1) % eval_ind == 0:
                test_model.eval()
                if "chimera" in args.data_path:
                    corrs = []
                    for b in test_dl:
                        t_out, _ = test_model.forward(b)
                        new_w = test_model.get_new_weights(batch, task="MLM").to(device)

                        indices = b['eval_indices']
                        sims = []
                        for probe in b['probe_inputs']:
                            p_sims = []
                            p_s = b['probe_inputs'][probe]['sentences']

                            for p_idx, s in enumerate(p_s):
                                enc = tokenizerTask.encode_plus(s[0], return_tensors='pt').to(device)
                                p_ind = indices[p_idx]
                                locs = get_locs(s[0], p_ind.item(), tokenizerTask)

                                p_emb = get_hidden_states(enc, locs, test_model.secondLM, [-1])
                                n_emb = get_emb(t_out.hidden_states, locs, [-1], p_idx)
                                p_sims.append(cos(n_emb, p_emb).item())

                            sim = sum(p_sims) / len(p_sims)
                            sims.append(sim)

                        ratings = [float(v) for v in b['ratings'][0].split(',')]
                        corr = stats.spearmanr(sims, ratings)
                        wandb.log({'test_point_correlation': corr.correlation})
                        corrs.append(corr.correlation)

                    test_model.memory.detach_past()
                    test_model.memory.memory = {}

                    avg_corr = sum(corrs) / len(corrs)
                    wandb.log({'epoch': epoch, 'Correlation on Test': avg_corr})

                    if avg_corr > best_corr:
                        chkpt_name = get_model_name_checkpoint(wandb.run.name, epoch)
                        save(test_model, opt, chkpt_name)
                        best_corr = avg_corr

                elif "sanity" in args.data_path:
                    test_model.eval()
                    test_losses = []
                    test_nonce_losses = []
                    for b in test_dl:
                        t_out, _ = test_model.forward(b)
                        wandb.log({'test point loss': t_out.loss.item()})
                        test_nonce_loss = get_nonce_loss(b, t_out, test_model.secondLM.vocab_size, device)
                        wandb.log({"test loss on nonce tokens": test_nonce_loss.item()})
                        test_nonce_losses.append(test_nonce_loss.item())

                        test_losses.append(t_out.loss.item())

                    wandb.log({'epoch': epoch, 'average test loss': sum(test_losses) / len(test_losses),
                               'average test nonce loss': sum(test_nonce_losses) / len(test_nonce_losses)})
                    n_loss = sum(test_nonce_losses) / len(test_nonce_losses)

                    if n_loss < best_loss:
                        chkpt_name = get_model_name_checkpoint(wandb.run.name, epoch)
                        save(test_model, opt, chkpt_name)
                        best_loss = n_loss

                elif "squad" in args.data_path:
                    test_model.eval()
                    test_losses = []
                    for b in test_dl:
                        t_out, _ = test_model.forward(b)

                        test_losses.append(t_out.loss.item())
                        test_model.memory.memory = {}

                    avg_test = sum(test_losses) / len(test_losses)
                    if avg_test < best_loss:
                        chkpt_name = get_model_name_checkpoint(wandb.run.name, epoch)
                        save(test_model, opt, chkpt_name)
                        best_loss = avg_test

                    wandb.log({'epoch': epoch, 'average test loss': avg_test})

                elif "snli" in args.data_path:
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



