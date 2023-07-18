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
    parser.add_argument("--cat", action="store_true")
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
        new_ids = swap_with_mask(task_inputs["input_ids"], k_task, nonceTask, mask_token_id, device)
    else:
        new_ids = task_inputs["input_ids"]
    task_ids = new_ids.reshape((b_task * k_task, l_task))  # reshape so we have n x seq_len
    task_attn = task_inputs["attention_mask"].reshape((b_task * k_task, l_task))

    return task_ids, task_attn, task_labels

class SimpleSNLIDatasetCat(Dataset):
    def __init__(self, data, tokenizerMLM, tokenizerTask, n_samples):
        #         super(SimpleSNLIDataset, self).__init__()
        self.premises = data["premise"]
        self.hypotheses = data["hypothesis"]
        self.sentences = data['sentences']
        self.replacements = data['replace']
        self.labels = data['label']
        self.n_samples = n_samples

        self.tokenizerMLM = tokenizerMLM
        self.tokenizerTask = tokenizerTask

        if not self.tokenizerTask.pad_token:
            self.tokenizerTask.pad_token = self.tokenizerTask.eos_token

    def __len__(self):
        return len(self.premises)

    def __getitem__(self, idx):
        premise = self.premises[idx]
        hypothesis = self.hypotheses[idx]
        replaced = self.replacements[idx]
        label = self.labels[idx]
        sentences = self.sentences[idx]

        if label == -1:
            return None

        nonce = snli_nonce(replaced.lower())

        premise = re.sub(r"\b({})\b".format(replaced), nonce, premise, flags=re.I)
        hypothesis = re.sub(r"\b({})\b".format(replaced), nonce, hypothesis, flags=re.I)

        nonceMLM = self.tokenizerMLM.convert_tokens_to_ids(nonce)
        nonceTask = self.tokenizerTask.convert_tokens_to_ids(nonce)

        do_sample = True

        if self.tokenizerMLM.model_max_length:
            mlm_length = self.tokenizerMLM.model_max_length
        else:
            raise Exception("Model Max Length does not exist for MLM")

        if self.tokenizerTask.model_max_length:
            task_length = self.tokenizerTask.model_max_length
        else:
            raise Exception("Model Max Length does not exist for TaskLM")

        #         sentences = [s.strip() for s in sentences]
        #         sentences = [s.replace('\n', ' ') for s in sentences]
        # #         print(replaced)
        # #         print(sentences)
        #         for s in sentences:
        #             if not re.search(r"\b({})\b".format(replaced), s):
        #                 return None

        sentences = [s.strip() for s in sentences]
        sentences = [s.replace('\n', " ") for s in sentences]
        sents = []

        if len(sentences) < self.n_samples:
            return None

        for s in sentences:
            if not re.search(r"\b({})\b".format(replaced), s, flags=re.I):
                continue
            else:
                sents.append(re.sub(r"\b({})\b".format(replaced), nonce, s, flags=re.I, count=1))

        sentences = sents

        try:
            if do_sample:
                sentences = np.random.choice(sentences, size=self.n_samples).tolist()
        except:
            return None

        tokensMLM = self.tokenizerMLM(sentences,
                                      max_length=mlm_length,
                                      padding='max_length',
                                      truncation=True,
                                      return_tensors='pt')

        modified_premise = " ".join(sentences + [premise])

        tokensTask = self.tokenizerTask(modified_premise,
                                        hypothesis,
                                        max_length=task_length,
                                        padding='max_length',
                                        truncation=True,
                                        return_tensors='pt',
                                        return_special_tokens_mask=True)

        return {
            'mlm_inputs': tokensMLM,  # shape for output is batch (per nonce) x k (examples) x 512 (tokens)
            'task_inputs': tokensTask,
            'nonceMLM': nonceMLM,
            'nonceTask': nonceTask,
            'task_labels': torch.LongTensor([label])
        }

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
    model = accelerator.prepare(model)

    if "snli" in args.data_path:
        n = args.num_examples
        split = dataset.train_test_split(0.5)
        # train = SimpleSNLIDataset(split["train"], tokenizer, tokenizer, n)
        if args.cat:
            test = SimpleSNLIDatasetCat(split["test"], tokenizer, tokenizer, n)
        else:
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
    accelerator.log({'average test accuracy mask': acc})

if __name__ == "__main__":
    main()
