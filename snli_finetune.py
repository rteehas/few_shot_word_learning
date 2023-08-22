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
from accelerate.utils import ProjectConfiguration
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, RobertaForMaskedLM, GPT2Model, RobertaForQuestionAnswering, \
    AutoModelForSequenceClassification, AutoModelForCausalLM, GPTJForCausalLM
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import load_from_disk

from configs.config import *
from eval_utils import compute_exact_match
from modules.buffer import RetrievalBuffer
from modules.model import MorphMemoryModel, MorphMemoryModelSQuAD, MorphMemoryModelSNLI, MorphMemoryModelGPT, \
    MorphMemoryModelGPTOnline, MorphMemoryModelGPTSubtoken, MorphMemoryModelMLMOnline, MorphMemoryModelGPTOnlineBinary, \
    MorphMemoryModelMLMOnlineBinary, MorphMemoryModelMLMOnlineFull
from data.data_utils import *
from train_utils import *
from data.few_shot_datasets import *


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-6)
    # parser.add_argument("--warmup", type=int, default=1e2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data_path", type=str, default="snli_with_generated_sentences")
    parser.add_argument("--tgt_data_path", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--memory", type=str, default="mean")
    parser.add_argument("--strategy", type=str, default='mask')
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--random_ex", action="store_true")
    parser.add_argument("--cat", action="store_true")
    parser.add_argument("--num_examples", type=int)
    parser.add_argument("--finetune", action="store_true")
    return parser

def load_model_partial(fname, model):
    model_dict = model.state_dict()
    state_dict = torch.load(fname)
    partial_states = {k:v for k, v in state_dict.items() if "emb_gen" in k or "memory" in k or "cls" in k or "dropout" in k}
#     print(partial_states)
    model_dict.update(partial_states)
    model.load_state_dict(model_dict)
    return model

def snli_nonce(word):
    return "<{}_nonce>".format(word)

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
    project = "snli_finetune"
    accelerator.init_trackers(
        project_name=project,
        config={"num_examples": args.num_examples,
                "finetuned": args.finetune
                })

    device = accelerator.device

    # mask_token_id = tokenizerMLM.mask_token_id

    test_model = MorphMemoryModelSNLI(firstLM, secondLM, list(set(new_toks)), device, layers,
                                      tokenizerMLM.mask_token_id, memory_config, emb_gen='Transformer').to(device)

    if args.finetune:
        path = "/scratch/rst306/few_shot_repo/model_checkpoints/resample_False_new_tok_warmup_mask_eval2_highWD_support_nodropout_fixed_finetuned_sentences_redo_full_gelu_online_6examples_1e-05_mean_Transformer_bs8_modified_mamlFalse_randomTrue_finetuneFalse_cat_Falselayers4_binary_False_mask_newTrue/MLMonline_memory_model_roberta_roberta_mean_memory/checkpoints/"
        chkpt = "checkpoint_22"
        chkpt_path = path + "{}/".format(chkpt)
        name = "pytorch_model.bin"

        test_model = load_model_partial(chkpt_path + name, test_model)

    n = args.num_examples

    split = dataset.train_test_split(0.2)
    train = SimpleSNLIDataset(split["train"], tokenizerMLM, tokenizerTask, n)
    test = SimpleSNLIDataset(split['test'], tokenizerMLM, tokenizerTask, n)

    train_dl = DataLoader(train, batch_size=args.batch_size, collate_fn=make_collate(train), shuffle=True, drop_last=True)
    test_dl = DataLoader(test, batch_size=args.batch_size, collate_fn=make_collate(test), drop_last=True)

    weight_decay = 0.02
    lr = 1e-5
    param_list = [{'params': filter(lambda p: p.requires_grad, test_model.emb_gen.parameters()), 'lr': lr,
                   'weight_decay': weight_decay}]

    opt = AdamW(param_list,
                eps=1e-8
                )

    warmup_steps = int(len(train_dl) * 0.3)
    eval_ind = 100

    epochs = args.epochs
    scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, epochs * len(train_dl))

    test_model, opt, train_dl, test_dl, scheduler = accelerator.prepare(
        test_model, opt, train_dl, test_dl, scheduler
    )

    for epoch in range(epochs):
        total_correct = 0
        total = 0
        for b in train_dl:
            log_dict = {}

            test_model.train()
            test_model.firstLM.eval()
            test_model.secondLM.eval()
            test_model.zero_grad()
            opt.zero_grad()


            test_model.process_memories(b['mlm_inputs'])
            t_out = test_model(b)
            preds = t_out.logits
            preds = F.log_softmax(preds, dim=-1).argmax(dim=1)
            true_ans = b['task_labels'].to(device).view(-1)
            num_correct = (preds == true_ans).sum()
            total_correct += num_correct
            total += b['task_labels'].shape[0]
            # all_losses = accelerator.gather(t_out.loss)
            log_dict['train loss'] = t_out.loss.item()
            accelerator.backward(t_out.loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(test_model.parameters(), 1.0)
            # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, test_model.parameters()), 1.0)
            for name, param in test_model.emb_gen.named_parameters():
                if param.grad is not None:
                    log_dict["post_{}_grad_norm".format(name)] = torch.norm(param.grad.view(-1)).item()
                    if torch.isnan(torch.norm(param.grad.view(-1))):
                        raise Exception("Nan Gradient")

            opt.step()
            scheduler.step()
            test_model.memory.memory = {}
            accelerator.log(log_dict)

        accelerator.log({"epoch": epoch, "average train acc": total_correct / total})

        #         true_fails[n] = []
        #     with open("transfer_failures.txt", 'a') as fp:
        #         fp.write("Writing Failures for n={} examples\n\n".format(n))
        with torch.no_grad():
            #         print("here")
            total_correct_test = 0
            total_test = 0
            for b in test_dl:
                test_model.process_memories(b['mlm_inputs'])
                t_out = test_model.forward(b)
                preds = t_out.logits
                preds = F.log_softmax(preds, dim=-1).argmax(dim=1)
                true_ans = b['task_labels'].to(device).view(-1)
                print("predicted", preds)
                print("true", true_ans)
                num_correct = (preds == true_ans).sum()
                total_correct_test += num_correct
                total_test += b['task_labels'].shape[0]
                #                     for ind in range(wrong.shape[0]):
                #     #                     with open("transfer_failures.txt", 'a') as fp:
                #     #                         fp.write("Premise/Hypothesis: {}\n".format(tokenizerTask.decode(wrong[ind, 0, :], skip_special_tokens=True)))
                #     #                         fp.write("True label: {}, Predicted Label: {}\n\n".format(true_ans[(preds != true_ans)][ind].item(),
                #     #                                                                               preds[(preds != true_ans)][ind].item()))
                #                         true_fails[n].append(true_ans[(preds != true_ans)][ind].item())
                test_model.memory.memory = {}
        # print("Accuracy for {} examples is: {}".format(n, acc))
        accelerator.log({"epoch": epoch, "average test accuracy".format(n): total_correct_test / total_test})
    accelerator.end_training()

if __name__ == "__main__":
    main()
