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
    parser.add_argument("--start_checkpoint", type=int)
    parser.add_argument("--end_checkpoint", type=int)
    return parser

def load_model_partial(fname, model):
    model_dict = model.state_dict()
    state_dict = torch.load(fname)
    partial_states = {k:v for k, v in state_dict.items() if "emb_gen" in k or "memory" in k or "cls" in k or "dropout" in k}
#     print(partial_states)
    model_dict.update(partial_states)
    model.load_state_dict(model_dict)
    return model

def main():
    args = get_arguments().parse_args()
    print("Arguments: ", args)

    layers = [-1, -2, -3, -4]  # add arg to pass this
    print("here")
    tokenizerMLM = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    print("made first tokenizer")
    firstLM = RobertaForMaskedLM.from_pretrained('roberta-base')
    print("made firstLM, tokenizer")

    secondLM = AutoModelForSequenceClassification.from_pretrained("./tmp_outputs2/snli_ft/checkpoint-137000")

    dataset = load_from_disk(args.data_path)
    nonces = list(map(snli_nonce, list(set(dataset['replacements']))))
    tokenizerTask = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)
    tokenizerMLM = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    tokenizerMLM.add_tokens(nonces)
    tokenizerTask.add_tokens(nonces)

    firstLM.resize_token_embeddings(len(tokenizerMLM))
    secondLM.resize_token_embeddings(len(tokenizerTask))

    memory_config = AggregatorConfig()
    new_toks = list(set(tokenizerMLM.convert_tokens_to_ids(nonces)))
    accelerator = Accelerator(log_with="wandb")
    project = "fewshot_model_snli"
    accelerator.init_trackers(
        project_name=project,
        config={"dataset": args.data_path})

    device = accelerator.device

    # mask_token_id = tokenizerMLM.mask_token_id

    # test_model = MorphMemoryModelSNLI(firstLM, secondLM, list(set(new_toks)), device, layers,
    #                                   tokenizerMLM.mask_token_id, memory_config, emb_gen='Transformer').to(device)

    path = "model_checkpoints/wd0.1_resample_False__redo_full_gelu_online_6examples_0.003_mean_Transformer_bs8_modified_mamlFalse_randomTrue_finetuneFalse_cat_Falselayers4_binary_False_mask_newTrue/MLMonline_memory_model_roberta_roberta_mean_memory_NormedOutput/checkpoints/"
    print(path)
    split = dataset.train_test_split(0.2)
    if "rescaleTrue" in path:
        rescale = True

    else:
        rescale = False

    print("Running SNLI Transfer eval with rescale = {}".format(rescale))
    for trial in range(5):
        for i in range(args.start_checkpoint, args.end_checkpoint + 1):
            chkpt = "checkpoint_{}".format(i)
            chkpt_path = path + "{}/".format(chkpt)
            name = "pytorch_model.bin"
            test_model = MorphMemoryModelSNLI(firstLM, secondLM, new_toks, device, [-1, -2, -3, -4],
                                              tokenizerMLM.mask_token_id, memory_config, 'Transformer', rescale).to(device)
            test_model = load_model_partial(chkpt_path + name, test_model)
            test_model = accelerator.prepare(test_model)
            test_model.eval()


            for n in range(1, args.num_examples + 1):
                # train = SimpleSNLIDataset(split["train"], tokenizerMLM, tokenizerTask, n)
                test = SimpleSNLIDataset(split['test'], tokenizerMLM, tokenizerTask, n)

                # train_dl = DataLoader(train, batch_size=5, collate_fn=make_collate(train), shuffle=True)
                test_dl = DataLoader(test, batch_size=args.batch_size, collate_fn=make_collate(test), drop_last=True)
                total_correct = 0
                total = 0
                test_dl = accelerator.prepare(test_dl)

                with torch.no_grad():
                    for b in test_dl:
                        t_out = test_model(b)
                        preds = t_out.logits
                        preds = F.log_softmax(preds, dim=-1).argmax(dim=1)
                        true_ans = b['task_labels'].to(device).view(-1)
                        num_correct = (preds == true_ans).sum()
                        total_correct += num_correct
                        total += b['task_labels'].shape[0]
                        test_model.memory.memory = {}

                acc = total_correct / total
                accelerator.log({"{}/{}_examples/average test accuracy".format(chkpt, n): acc})

if __name__ == "__main__":
    main()
