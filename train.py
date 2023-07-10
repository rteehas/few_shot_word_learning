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
from eval_utils import compute_exact_match
from modules.buffer import RetrievalBuffer
from modules.model import MorphMemoryModel, MorphMemoryModelSQuAD, MorphMemoryModelSNLI, MorphMemoryModelGPT, \
    MorphMemoryModelGPTOnline, MorphMemoryModelGPTSubtoken, MorphMemoryModelMLMOnline
from data.data_utils import *
from train_utils import *
from data.few_shot_datasets import *

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-6)
    # parser.add_argument("--warmup", type=int, default=1e2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--tgt_data_path", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load_checkpoint", type=bool, default=False)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--taskName", type=str, default='mlm')
    parser.add_argument("--secondLM", type=str)
    parser.add_argument("--memory", type=str, default="mean")
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--intermediate_loss", type=bool, default=False)
    parser.add_argument("--trial", type=str, default='l2')
    parser.add_argument("--emb_gen", type=str, default='mlp')
    parser.add_argument("--strategy", type=str, default='mask')
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--maml", action="store_true")
    parser.add_argument("--word_path", type=str, default='')
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--random_ex", action="store_true")
    return parser


def main():
    args = get_arguments().parse_args()

    print("Arguments: ", args)
    accelerator = Accelerator()
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = accelerator.device
    layers = [-1] # add arg to pass this
    print("here")
    tokenizerMLM = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    print("made first tokenizer")
    firstLM = RobertaForMaskedLM.from_pretrained('roberta-base')
    print("made firstLM, tokenizer")
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
        secondLM = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-roberta-base')
    elif args.taskName == "addition":
        tokenizerTask = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
        tokenizerTask.pad_token = tokenizerTask.unk_token
        secondLM = AutoModelForCausalLM.from_pretrained("gpt2")
        nonces = ["<OP>"]
        dataset_name = "addition"
    elif args.taskName == "addition_subtok":
        tokenizerTask = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
        tokenizerTask.pad_token = tokenizerTask.unk_token
        secondLM = AutoModelForCausalLM.from_pretrained("gpt2")
        nonces = ["<NUM1>", "<NUM2>"]
        dataset_name = "addition_subtok"

    elif args.taskName == "online":
        if args.secondLM == "gpt2":
            tokenizerTask = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
            tokenizerTask.pad_token = tokenizerTask.unk_token
            print("making model")
            secondLM = AutoModelForCausalLM.from_pretrained("gpt2")
            print("made model")
        elif args.secondLM == "roberta":
            tokenizerTask = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)
            secondLM = RobertaForMaskedLM.from_pretrained('roberta-base')
        else:
            raise NotImplementedError("{} Not Implemented for Task LM".format(args.secondLM))
        # with open(args.word_path, 'r') as f:
        #     reader = csv.reader(f)
        #     rows = [row for row in reader]
        # words = rows[0]
        word_dict = load_from_disk(args.word_path)
        words = word_dict['train']['words'] + word_dict['test']['words']
        nonces = list(map(lambda w: "<{}_new>".format(w), words))
    else:
        raise NotImplementedError("{} not implemented".format(args.taskName))

    if "addition" not in args.taskName:
        dataset = load_from_disk(args.data_path)
        # if args.taskName == "online":
        #     dataset = dataset.filter(lambda ex: len(ex['text']) > 10)

    print("here2")
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
        weight_decay = 0.02

    elif args.memory == "rnn":
        memory_config = RNNAggConfig()
        weight_decay = 0.015
    elif args.memory == "cls":
        memory_config = TransformerCLSConfig()
        weight_decay = 0.015
    else:
        raise NotImplementedError("This memory aggregation is not implemented")

    mask_token_id = tokenizerMLM.mask_token_id

    if "chimera" in args.data_path:

        cos = nn.CosineSimilarity(dim=-1)

        split = dataset["train"].train_test_split(test_size=0.2)
        mlm_dataset = ChimeraMLMDataset(split["train"], tokenizerMLM, tokenizerTask, args.num_examples, args.trial)

        train_dl = DataLoader(mlm_dataset, batch_size=args.batch_size, shuffle=True)

        test_dataset = ChimeraTestDataset(split["test"], tokenizerMLM, tokenizerTask, args.num_examples, args.trial)

        collate = make_collate(test_dataset)

        test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    elif "sanity" in args.data_path:
        split = dataset.train_test_split(test_size=0.2)
        n = args.num_examples
        tokenizerMLM.add_tokens(nonces)
        tokenizerTask.add_tokens(nonces)

        mlm_dataset = SimpleMLMDataset(split["train"], tokenizerMLM, tokenizerTask, n)

        train_dl = DataLoader(mlm_dataset, batch_size=args.batch_size, shuffle=True)

        # train_eval = ChimeraTestDataset(chimera["train"], tokenizerMLM, tokenizerTask, n, trial)

        test_dataset = SimpleMLMDataset(split["test"], tokenizerMLM, tokenizerTask, n)

        # collate = make_collate(test_dataset)

        test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    elif "squad" in args.data_path:

        tokenizerMLM.add_tokens(nonces)
        tokenizerTask.add_tokens(nonces)

        split = dataset.train_test_split(0.2)

        train = SimpleSQuADDataset(split['train'], tokenizerMLM, tokenizerTask, args.num_examples)

        test = SimpleSQuADDataset(split['test'], tokenizerMLM, tokenizerTask, args.num_examples)

        train_dl = DataLoader(train, batch_size=args.batch_size, collate_fn=make_collate(train), shuffle=True)

        test_dl = DataLoader(test, batch_size=args.batch_size, collate_fn=make_collate(test))

    elif "snli" in args.data_path:
        n = args.num_examples
        split = dataset.train_test_split(0.2)
        train = SimpleSNLIDataset(split["train"], tokenizerMLM, tokenizerTask, n)
        test = SimpleSNLIDataset(split["test"], tokenizerMLM, tokenizerTask, n)

        train_dl = DataLoader(train, batch_size=args.batch_size, collate_fn=make_collate(train), shuffle=True)
        test_dl = DataLoader(test, batch_size=args.batch_size, collate_fn=make_collate(test))
    elif "wikitext" in args.data_path:
        # split = dataset.train_test_split(0.2)

        train = SimpleOnlineDataset(dataset['train'], tokenizerMLM, tokenizerTask)
        train_dl = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)

        test = SimpleOnlineDataset(dataset['test'], tokenizerMLM, tokenizerTask)
        test_dl = DataLoader(test, batch_size=args.batch_size, shuffle=True, drop_last=True)

    else:
        if args.taskName == "addition":
            train_size = 10000
            operation = "addition"
            orthography = "decimal"
            base_number = 10
            min_digits_train, max_digits_train = 2, 5
            train = MyDataset(n_examples=train_size, min_digits=min_digits_train,
                              max_digits=max_digits_train,
                              operation=operation, orthography=orthography,
                              base_number=base_number, invert_question=False,
                              invert_answer=False, balance=True)

            test = MyDataset(n_examples=1000, min_digits=min_digits_train,
                             max_digits=max_digits_train,
                             operation=operation, orthography=orthography,
                             base_number=base_number, invert_question=False,
                             invert_answer=False, balance=True)

            train_set = SimpleMathDataset(train, tokenizerMLM, tokenizerTask, 30)
            train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)

            test_set = SimpleMathDataset(test, tokenizerMLM, tokenizerTask, None)

            test_dl = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
        elif args.taskName == "addition_subtok":
            train_size = 10000
            operation = "addition"
            orthography = "decimal"
            base_number = 10
            min_digits_train, max_digits_train = 2, 5
            train = MyDataset(n_examples=train_size, min_digits=min_digits_train,
                              max_digits=max_digits_train,
                              operation=operation, orthography=orthography,
                              base_number=base_number, invert_question=False,
                              invert_answer=False, balance=True)

            test = MyDataset(n_examples=1000, min_digits=min_digits_train,
                             max_digits=max_digits_train,
                             operation=operation, orthography=orthography,
                             base_number=base_number, invert_question=False,
                             invert_answer=False, balance=True)

            train_set = SimpleMathDatasetSubtok(train, tokenizerMLM, tokenizerTask, 30)
            train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)

            test_set = SimpleMathDatasetSubtok(test, tokenizerMLM, tokenizerTask, None)

            test_dl = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)

        else:
            raise NotImplementedError

    new_toks = tokenizerMLM.convert_tokens_to_ids(nonces)


    epochs = args.epochs
    lr = args.lr
    epsilon = 1e-8


    if "squad" in args.data_path:
        test_model = MorphMemoryModelSQuAD(firstLM, secondLM, new_toks,
                                      device, layers, mask_token_id, memory_config, args.emb_gen)
    elif "snli" in args.data_path:
        test_model = MorphMemoryModelSNLI(firstLM, secondLM, new_toks, device, [-1],
                                   tokenizerMLM.mask_token_id, memory_config, args.emb_gen)
    elif "sanity" in args.data_path:
        test_model = MorphMemoryModel(firstLM, secondLM, new_toks,
                                  device, layers, mask_token_id, memory_config, args.emb_gen)
    elif "wikitext" in args.data_path:
        buffer = RetrievalBuffer(15, args.num_examples, new_toks, tokenizerMLM, args.random_ex)
        if args.secondLM == "gpt2":
            test_model = MorphMemoryModelGPTOnline(firstLM, secondLM, new_toks, device, [-1],
                                                   tokenizerMLM.mask_token_id, memory_config, emb_type='Transformer')
        elif args.secondLM == "roberta":
            test_model = MorphMemoryModelMLMOnline(firstLM, secondLM, new_toks, device, [-1],
                                                   tokenizerMLM.mask_token_id, memory_config,
                                                   emb_type='Transformer')
    else:
        if args.taskName == "addition":
            test_model = MorphMemoryModelGPT(firstLM, secondLM, new_toks, device, [-1],
                                             tokenizerMLM.mask_token_id, memory_config, emb_type='Transformer')
        elif args.taskName == "addition_subtok":
            test_model = MorphMemoryModelGPTSubtoken(firstLM, secondLM, new_toks, device, [-1, -2],
                                             tokenizerMLM.mask_token_id, memory_config, emb_type='Transformer')
        else:
            raise NotImplementedError
    if args.finetune:
        test_model.secondLM.get_input_embeddings().weight.requires_grad = True

        param_list = [{"params": filter(lambda p: p.requires_grad, test_model.secondLM.parameters()), 'lr': 1e-5},
                      {'params': filter(lambda p: p.requires_grad, test_model.emb_gen.parameters()), 'lr': lr, 'weight_decay': weight_decay}]
    else:
        param_list = [{'params': filter(lambda p: p.requires_grad, test_model.emb_gen.parameters()), 'lr': lr,
                       'weight_decay': weight_decay}]

    opt = AdamW(param_list,
                eps=epsilon
                )

    warmup_steps = len(train_dl)
    eval_ind = 100
    if args.taskName == "addition":
        eval_ind=30

    scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, epochs * len(train_dl))
    intermediate = args.intermediate_loss

    test_model
    test_model, opt, train_dl, test_dl, scheduler = accelerator.prepare(
        test_model, opt, train_dl, test_dl, scheduler
    )

    project = "fewshot_model_{}".format(args.taskName)

    run = wandb.init(project=project, reinit=True)
    wandb.run.name = "gelu_{}_{}examples_{}_{}_{}_bs={}_modified_maml={}_random={}_finetune={}".format(dataset_name,
                                                                            args.num_examples,
                                                                            lr,
                                                                            memory_config.agg_method,
                                                                            args.emb_gen,
                                                                            args.batch_size,
                                                                            args.maml,
                                                                            args.random_ex,
                                                                            args.finetune)

    if intermediate:
        wandb.run.name = wandb.run.name + "_intermediate"

    os.makedirs("/scratch/rst306/few_shot_word_learning/checkpoints/{}".format(dataset_name), exist_ok=True)
    os.makedirs("/scratch/rst306/few_shot_word_learning/checkpoints/{}/{}".format(dataset_name, wandb.run.name.replace("=", "")), exist_ok=True)

    save_folder = "{}/{}/".format(dataset_name, wandb.run.name)

    best_corr = 0
    best_acc = 0
    best_loss = 10000
    n_inner_iter = 3
    eval_num = 100
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

                if args.taskName == "online":

                    to_sample = [n for n in buffer.nonces if n in batch['mlm_inputs']['input_ids']]
                    for n in to_sample:
                        sample = buffer.retrieve(n)
                        if sample is not None:
                            test_model.process_memories(sample)

                out, losses = test_model(batch)

                loss = out.loss

                log_dict['train loss'] = loss.item()
                train_losses.append(loss.item())

            elif args.maml:
                inner_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, test_model.parameters()),
                                            lr=1e-5)

                test_model.store_mem(batch)
                with higher.innerloop_ctx(
                        test_model, inner_opt, copy_initial_weights=False
                ) as (fnet, diffopt):
                    inner_losses = []
                    for _ in range(n_inner_iter):
                        maml_out = test_model.forward_inner(batch)

                        inner_losses.append(maml_out.loss.item())
                        diffopt.step(maml_out.loss)

                    out, l = test_model(batch)
                    log_dict['average maml inner loss'] = sum(inner_losses) / len(inner_losses)
                    log_dict['maml outer loss'] =  out.loss.item()
                    #out.loss.backward()
                    loss = out.loss

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

            # final_loss.backward()
            accelerator.backward(final_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(test_model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, test_model.parameters()), 1.0)
            # for name, param in test_model.emb_gen.named_parameters():
            #     if param.grad is not None:
            #         log_dict["post_{}_grad_norm".format(name)] = torch.norm(param.grad.view(-1)).item()
            #         if torch.isnan(torch.norm(param.grad.view(-1))):
            #             raise Exception("Nan Gradient")

            # if args.taskName == "addition":
                # for ind, val in enumerate(batch['generationTokens']):
                #     idx = deepcopy(val)
                #     gen_ans = test_model.generate(idx, 10)
                #     gen_ans = tokenizerTask.decode(gen_ans['input_ids'][0], skip_special_tokens=True,
                #                                    clean_up_tokenization_spaces=True)
                #     true_ans = tokenizerTask.decode(batch['task_inputs']['input_ids'][ind, 0, :], skip_special_tokens=True,
                #                                     clean_up_tokenization_spaces=True)
                #     train_total += 1
                #     train_correct += compute_exact_match(gen_ans, true_ans)

            opt.step()
            scheduler.step()
            with torch.no_grad():
                for v in test_model.memory.memory:
                    for val in test_model.memory.memory[v]:
                        del val
            test_model.memory.memory = {}
            if args.taskName == "online":
                buffer.store(batch['mlm_inputs'].to(device))
                buffer.cleanup()
            wandb.log(log_dict)

            if i != 0 and (i % eval_ind == 0 or i % len(train_dl) == 0):
                opt.zero_grad(set_to_none=True)
                test_model.eval()
                with torch.no_grad():
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
                            chkpt_name = get_model_name_checkpoint(save_folder + test_model.model_name, epoch)
                            print(chkpt_name)
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
                            chkpt_name = get_model_name_checkpoint(save_folder + test_model.model_name, epoch)
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
                            chkpt_name = get_model_name_checkpoint(save_folder + test_model.model_name, epoch)
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
                            with torch.no_grad():
                                for v in test_model.memory.memory:
                                    for val in test_model.memory.memory[v]:
                                        del val
                            test_model.memory.memory = {}
                        acc = total_correct / total
                        wandb.log({'epoch': epoch, 'average test accuracy': acc})
                        if best_acc < acc:
                            chkpt_name = get_model_name_checkpoint(save_folder + test_model.model_name, epoch)
                            save(test_model, opt, chkpt_name)
                            print("Saved {}".format(chkpt_name))
                            best_acc = acc

                    elif "addition" in args.taskName:
                        test_matches = 0
                        test_total = 0
                        test_losses = []
                        for b in test_dl:
                            t_out, _ = test_model.forward(b)

                            test_losses.append(t_out.loss.item())

                            for ind, val in enumerate(b['generationTokens']):
                                idx = deepcopy(val)
                                gen_ans = test_model.generate(idx, 10)
                                gen_ans = tokenizerTask.decode(gen_ans['input_ids'][0], skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=True)
                                true_ans = tokenizerTask.decode(b['task_inputs']['input_ids'][ind, 0, :],
                                                                skip_special_tokens=True, clean_up_tokenization_spaces=True)
                                test_total += 1
                                test_matches += compute_exact_match(gen_ans, true_ans)

                            with torch.no_grad():
                                for v in test_model.memory.memory:
                                    for val in test_model.memory.memory[v]:
                                        del val
                            test_model.memory.memory = {}
                        avg_test = sum(test_losses) / len(test_losses)
                        avg_match = test_matches / test_total
                        wandb.log({'epoch': epoch, 'average test loss': avg_test, "test exact match": avg_match})
                        if avg_test < best_loss:
                            chkpt_name = get_model_name_checkpoint(save_folder + test_model.model_name, epoch)
                            save(test_model, opt, chkpt_name)
                            print("Saved {}".format(chkpt_name))
                            best_loss = avg_test

                    elif args.taskName == "online":
                        test_model.eval()
                        test_losses = []
                        for b in test_dl:
                            to_sample = [n for n in buffer.nonces if n in b['mlm_inputs']['input_ids']]
                            for n in to_sample:
                                sample = buffer.retrieve(n)
                                if sample is not None:
                                    test_model.process_memories(sample)
                            t_out, _ = test_model.forward(b)

                            test_losses.append(t_out.loss.item())
                            test_model.memory.memory = {}
                        avg_test = sum(test_losses) / len(test_losses)
                        wandb.log({'epoch': epoch, 'average test loss': avg_test})

                        if avg_test < best_loss:
                            chkpt_name = get_model_name_checkpoint(save_folder + test_model.model_name, eval_ind)
                            print(chkpt_name)
                            save(test_model, opt, chkpt_name)
                            print("Saved {}".format(chkpt_name))
                            best_loss = avg_test

        wandb.log({"epoch": epoch, 'average train loss': sum(train_losses) / len(train_losses)})
        if "snli" in args.data_path:
            wandb.log({"epoch": epoch, 'average train acc': train_correct / train_total})
        # if args.taskName == "addition":
        #     wandb.log({'epoch': epoch,
        #                'train exact match': train_correct / train_total})

if __name__ == "__main__":
    main()
