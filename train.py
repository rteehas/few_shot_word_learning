from transformers import DataCollatorForLanguageModeling
from argparse import ArgumentParser
import wandb
import torch
import torch.nn as nn
import math
import evaluate
from transformers import AutoTokenizer, RobertaTokenizerFast
from configs.config import Config
from modules.model import MorphMemoryModel
from data.data_utils import get_nonces, get_basic_dataset

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--lr_weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--tgt_data_path", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load_checkpoint", type=bool, default=False)
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    return parser

def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def train(model, train_set, config, tokenizer, device):
    lr = config.lr
    mlm_prob = config.mlm_prob
    weight_decay = config.weight_decay
    epochs = config.epochs

    mlm_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=mlm_prob)
    accuracy_metric = evaluate.load("accuracy")

    wandb.init(project="initial-fewshot-run")
    run = wandb.init(project="initial-fewshot-run", reinit=True)
    wandb.run.name = "run_lr={0}_mlm={1}_norm_clipping_1.0_layer4_weightdecay".format(lr, mlm_prob)
    opt = torch.optim.AdamW(model.params(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    k = 5
    accs = []
    topk_accs = []
    for i in range(epochs):
        for j, row in enumerate(train_set):
            opt.zero_grad()
            seq = row[1]
            second_seq = row[2]
            nonce = row[-1]

            seq_toks = tokenizer(seq, return_tensors="pt")
            second_seq_toks = tokenizer(second_seq, return_tensors="pt")
            # print(seq_toks)
            # seq_ids, seq_labels = mlm_collator.torch_mask_tokens(seq_toks["input_ids"])
            second_ids, second_labels = mlm_collator.torch_mask_tokens(second_seq_toks["input_ids"])

            # seq_inputs = {"input_ids": seq_ids.to(device), "attention_mask": seq_toks["attention_mask"].to(device)}
            seq_inputs = seq_toks.to(device)
            second_inputs = {"input_ids": second_ids.to(device),
                             "attention_mask": second_seq_toks["attention_mask"].to(device)}

            loss, hiddens, logits = model(seq, second_seq, seq_inputs, second_inputs, second_labels.to(device), nonce)

            preds = torch.argmax(logits, dim=-1)
            msk_idx = torch.where(second_inputs["input_ids"] == model.tokenizer.mask_token_id)[1]
            true_labels = second_labels.to(device)[:, msk_idx].tolist()
            pred_labels = preds[:, msk_idx].tolist()
            acc = accuracy_metric.compute(references=true_labels[0], predictions=pred_labels[0])
            if not math.isnan(acc["accuracy"]):
                accs.append(acc["accuracy"])
                wandb.log({"Accuracy": acc["accuracy"]})

            topk_preds = torch.topk(logits[:, msk_idx, :], k, dim=-1).indices[0].tolist()
            top_pred = []
            for true, tpk in zip(true_labels[0], topk_preds):
                if true in tpk:
                    top_pred.append(true)
                else:
                    top_pred.append(-100)

            topk_acc = accuracy_metric.compute(references=true_labels[0], predictions=top_pred)
            if not math.isnan(topk_acc["accuracy"]):
                topk_accs.append(topk_acc["accuracy"])
                wandb.log({"Top {} Accuracy".format(k): topk_acc["accuracy"]})

            loss.backward()
            first_norm = get_grad_norm(model)
            wandb.log({"Grad Norm before clip": first_norm})

            nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
            post_clip = get_grad_norm(model)
            wandb.log({"Grad Norm after clipping": post_clip})

            opt.step()
            # loss_log = {"first_lm_loss": losses[0], "masked_step_loss": losses[1], "second_lm_loss": losses[2]}
            # print(loss_log)
            print(loss)
            wandb.log({"Loss": loss})
            if ((i + 1) * (j + 1)) % 100 == 0:

                if len(accs) > 0:
                    # val_accs = accs[-100:]
                    avg_acc = sum(accs[-100:]) / len(accs[-100:])
                    avg_topk = sum(topk_accs[-100:]) / len(topk_accs[-100:])
                    # print(accs)
                    # print(topk_accs)
                    wandb.log({"Average MLM Accuracy": avg_acc})
                    wandb.log({"Average Top {} Accuracy".format(k): avg_topk})

    run.finish()

if __name__ == "__main__":
    parser = get_arguments()
    args, _ = parser.parse_known_args()
    config = Config
    config.lr = args.lr
    config.weight_decay = args.weight_decay
    device = args.device

    nonce_toks = get_nonces(args.data_path)

    train_set = get_basic_dataset(args.data_path, args.tgt_data_path)

    layers = [-4, -3, -2, -1]
    tokenizer = AutoTokenizer.from_pretrained("roberta-large", use_fast=True)
    tokenizer.add_tokens(nonce_toks)

    model = MorphMemoryModel(tokenizer, nonce_toks, device, layers=layers).to(device)

    train(model,train_set, config, tokenizer, device)