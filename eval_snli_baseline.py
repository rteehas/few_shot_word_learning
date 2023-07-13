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
    parser.add_argument("--percent_first", type=float, default=0.0)
    # parser.add_argument("--warmup", type=int, default=1e2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--model_dir", type=str, default="../few_shot_word_learning/checkpoints/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--taskName", type=str, default='snli')
    parser.add_argument("--num_examples", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--word_path", type=str, default='')
    return parser


def swap_with_mask(inputs, k_examples, nonces, mask_token_id, device):
    inp = inputs.clone()
    exp = torch.Tensor(nonces).unsqueeze(1).expand_as(inputs[:, 0, :]).to(
        device)  # expand for a set of sentences across batches
    exp = exp.unsqueeze(1).repeat(1, k_examples, 1)  # repeat for k sentences
    inp[inp == exp] = mask_token_id

    return inp


def prepare_baseline_batch(batch, mask_token_id, device, use_mask):
    task_inputs = {'input_ids': batch["task_inputs"]['input_ids'].to(device),
                   'attention_mask': batch["task_inputs"]['attention_mask'].to(device)}
    nonceTask = batch["nonceTask"]
    if 'task_labels' in batch:
        task_labels = batch["task_labels"].to(device)

    b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape
    if use_mask:
        new_ids = swap_with_mask(task_inputs["input_ids"], k_task, nonceTask, mask_token_id)
    else:
        new_ids = task_inputs["input_ids"]
    task_ids = new_ids.reshape((b_task * k_task, l_task))  # reshape so we have n x seq_len
    task_attn = task_inputs["attention_mask"].reshape((b_task * k_task, l_task))

    return task_ids, task_attn, task_labels


def main():
    args = get_arguments().parse_args()
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    dataset = load_from_disk(args.data_path)
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    if "snli" in args.data_path:
        nonces = list(map(snli_nonce, list(set(dataset['replace']))))
        dataset_name = "snli"
    else:
        raise NotImplementedError

    tokenizer.add_tokens(nonces)
    model.resize_token_embeddings(len(tokenizer))

    accelerator = Accelerator(log_with="wandb")
    device = accelerator.device

    if "snli" in args.data_path:
        n = args.num_examples
        split = dataset.train_test_split(0.5)
        # train = SimpleSNLIDataset(split["train"], tokenizer, tokenizer, n)
        test = SimpleSNLIDataset(split["test"], tokenizer, tokenizer, n)

        # train_dl = DataLoader(train, batch_size=args.batch_size, collate_fn=make_collate(train), shuffle=True)
        test_dl = DataLoader(test, batch_size=args.batch_size, collate_fn=make_collate(test))

    project = "snli_eval_baselines"
    accelerator.init_trackers(
        project_name=project,
        config={"num_examples": args.num_examples,
                },
        # init_kwargs={"wandb": {"name": run_name}},
    )

    model.eval()
    total_correct = 0
    total = 0
    for b in test_dl:
        with torch.no_grad():
            ids, attn, labels = prepare_baseline_batch(b, tokenizer.mask_token_id, device, use_mask=False)
            t_out = model(input_ids=ids,
                             attention_mask=attn,
                             labels=labels.squeeze(1))
            preds = t_out.logits
            preds = F.log_softmax(preds, dim=-1).argmax(dim=1)
            true_ans = b['task_labels'].to(device).view(-1)
            num_correct = (preds == true_ans).sum()
            total_correct += num_correct
            total += b['task_labels'].shape[0]
    acc = total_correct / total
    accelerator.log({'average test accuracy no mask': acc})

    total_correct = 0
    total = 0
    for b in test_dl:
        with torch.no_grad():
            batch = prepare_baseline_batch(b, tokenizer.mask_token_id, device, use_mask=True)
            t_out = model.forward(batch)
            preds = t_out.logits
            preds = F.log_softmax(preds, dim=-1).argmax(dim=1)
            true_ans = batch['task_labels'].to(device).view(-1)
            num_correct = (preds == true_ans).sum()
            total_correct += num_correct
            total += batch['task_labels'].shape[0]

    acc = total_correct / total
    accelerator.log({'average test accuracy mask': acc})

if __name__ == "__main__":
    main()
