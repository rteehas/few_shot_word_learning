import math
from argparse import ArgumentParser
from functools import reduce
import os
from copy import deepcopy
from scipy import stats

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaForMaskedLM, GPT2Model
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_from_disk

from configs.config import *
from modules.model import MorphMemoryModel
from data.data_utils import *
from train_utils import *
from data.few_shot_datasets import *

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
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

    if args.taskName == "autoregressive":
        tokenizerTask = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
        secondLM = GPT2Model.from_pretrained("gpt2")

    dataset = load_from_disk(args.data_path)

    if "chimera" in args.data_path:
        nonces = list(map(make_nonce, list(set(dataset["train"]['id'] + dataset["test"]['id']))))
        dataset_name = "chimera{}".format(args.num_examples)

    if "sanity" in args.data_path:
        nonces = list(map(make_sanity_nonce, list(set(dataset['word']))))
        dataset_name = "sanity"
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

    mask_token_id = tokenizerTask.mask_token_id

    if "chimera" in args.data_path:

        cos = nn.CosineSimilarity(dim=-1)

        split = dataset["train"].train_test_split(test_size=0.2)
        mlm_dataset = ChimeraMLMDataset(split["train"], tokenizerMLM, tokenizerTask, args.num_examples, args.trial)

        mlm_dataloader = DataLoader(mlm_dataset, batch_size=1, shuffle=True)

        test_dataset = ChimeraTestDataset(split["test"], tokenizerMLM, tokenizerTask, args.num_examples, args.trial)

        collate = make_collate(test_dataset)

        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate)

    if "sanity" in args.data_path:
        split = dataset.train_test_split(test_size=0.2)
        n = args.num_examples



        tokenizerMLM.add_tokens(nonces)
        tokenizerTask.add_tokens(nonces)

        mlm_dataset = SimpleMLMDataset(split["train"], tokenizerMLM, tokenizerTask, n)

        mlm_dataloader = DataLoader(mlm_dataset, batch_size=1, shuffle=True)

        # train_eval = ChimeraTestDataset(chimera["train"], tokenizerMLM, tokenizerTask, n, trial)

        test_dataset = SimpleMLMDataset(split["test"], tokenizerMLM, tokenizerTask, n)

        # collate = make_collate(test_dataset)

        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    nonces = tokenizerTask.convert_tokens_to_ids(nonces)


    epochs = args.epochs
    lr = args.lr
    epsilon = 1e-8

    test_model = MorphMemoryModel(firstLM, secondLM, nonces,
                                  device, layers, mask_token_id, memory_config).to(device)

    opt = AdamW(filter(lambda p: p.requires_grad, test_model.parameters()),
                lr=lr,
                eps=epsilon
                )

    intermediate = args.intermediate_loss

    run = wandb.init(project="fewshot_model_testing_redone", reinit=True)
    wandb.run.name = "{}_{}_{}_intermediate={}".format(dataset_name, lr, memory_config.agg_method, intermediate)



    best_corr = 0
    best_loss = 10000
    for epoch in range(epochs):
        train_corr = []
        for i, batch in enumerate(mlm_dataloader):
            #         print(i)
            test_model.train()

            test_model.zero_grad()
            opt.zero_grad()
            out, losses = test_model(batch)

            loss = out.loss

            wandb.log({'train mlm loss': loss.item()})

            nonce_loss = get_nonce_loss(batch, out, test_model.secondLM.vocab_size, device)
            if nonce_loss:
                wandb.log({"new token loss": nonce_loss.item()})

            if intermediate:
                intermediate_loss = torch.sum(torch.Tensor([losses]))
                wandb.log({"emb generator loss": intermediate_loss.item()})
                final_loss = loss + intermediate_loss
            else:
                final_loss = loss

            final_loss.backward()
            opt.step()
            test_model.memory.detach_past()

        test_model.eval()
        if "chimera" in args.data_path:
            corrs = []
            for b in test_dataloader:
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
            wandb.log({'Correlation on Test': avg_corr})

            if avg_corr > best_corr:
                chkpt_name = get_model_name_checkpoint(wandb.run.name, epoch)
                save(test_model, opt, chkpt_name)
                best_corr = avg_corr

        if "sanity" in args.data_path:
            test_model.eval()
            test_losses = []
            test_nonce_losses = []
            for b in test_dataloader:
                t_out, _ = test_model.forward(b)
                wandb.log({'test point loss': t_out.loss.item()})
                test_nonce_loss = get_nonce_loss(b, t_out, test_model.secondLM.vocab_size, device)
                wandb.log({"test loss on nonce tokens": test_nonce_loss.item()})
                test_nonce_losses.append(test_nonce_loss.item())

                test_losses.append(t_out.loss.item())

            wandb.log({'average test loss': sum(test_losses) / len(test_losses)})
            wandb.log({'average test nonce loss': sum(test_nonce_losses) / len(test_nonce_losses)})
            n_loss = sum(test_nonce_losses) / len(test_nonce_losses)

            if n_loss < best_loss:
                chkpt_name = get_model_name_checkpoint(wandb.run.name, epoch)
                save(test_model, opt, chkpt_name)
                best_loss = n_loss

            test_model.memory.detach_past()
            test_model.memory.memory = {}

