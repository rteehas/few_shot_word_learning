import torch
from torch import nn
import numpy as np
import os

def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def save(model, opt, model_name_chkpt, accelerator):
    # save
    fname = "/scratch/rst306/few_shot_word_learning/checkpoints/{}".format(model_name_chkpt)

    i =0

    save_name = fname + "_{}".format(i)
    while os.path.exists(save_name):
        i+=1
        save_name = fname + "_{}".format(i)

    save_name = save_name+".pth"
    accelerator.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict()},
        save_name)


def get_hidden_states(encoded, token_ids_word, model, layers):
    with torch.no_grad():
        output = model(**encoded, output_hidden_states=True)

    states = output.hidden_states

    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    word_tokens_output = output[token_ids_word]

    return word_tokens_output.mean(dim=0)


def get_emb(hidden, locs, layers, idx):
    output = torch.stack([hidden[i][idx, :, :] for i in layers]).sum(0).squeeze()

    word_tokens_output = output[locs]
    return word_tokens_output.mean(dim=0)


def get_per_token_loss(labels, logits, nonces, vocab_size):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))

    selected = loss[torch.where(torch.isin(torch.flatten(labels), nonces))]

    return selected

def get_new_token_loss_labels(labels, logits, vocab_size, new_tokens):
    token_loss = get_per_token_loss(labels, logits, new_tokens, vocab_size)
    # print(token_loss)
    if token_loss.numel() > 0:
        return token_loss.mean()
    else:
        return None


def get_per_token_loss_llama(labels, logits, nonces, vocab_size):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)

    loss = loss_fct(shift_logits, shift_labels)

    selected = loss[torch.where(torch.isin(torch.flatten(shift_labels), nonces))]

    return selected


def get_new_token_loss_labels_llama(labels, logits, vocab_size, new_tokens):
    token_loss = get_per_token_loss_llama(labels, logits, new_tokens, vocab_size)
    # print(token_loss)
    if token_loss.numel() > 0:
        return token_loss.mean()
    else:
        return None


def get_nonce_loss(batch, out, vocab_size, new_tokens, device):
    b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape

    # nonceTask = batch['nonceTask'].to(device)
    task_labels = batch["task_labels"].to(device).reshape((b_task * k_task, l_task))

    token_loss = get_per_token_loss(task_labels, out.logits, new_tokens, vocab_size)
    if token_loss.numel() > 0:
        return token_loss.mean()
    else:
        return None
def get_new_token_loss_internal(batch, logits, vocab_size, new_tokens):
    b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape
    task_labels = batch["task_labels"].to(logits.device).reshape((b_task * k_task, l_task))
    token_loss = get_per_token_loss(task_labels, logits, new_tokens, vocab_size)
    if token_loss.numel() > 0:
        return token_loss.mean()
    else:
        return None

def get_model_name_checkpoint(run_name, epoch):
    return "{}_epoch={}".format(run_name, epoch)

def get_word_idx(sent, word):
    return sent.split(" ").index(word)

def get_locs(sent, idx, tokenizer):
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")
    token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
    return token_ids_word

def snli_nonce(word):
    return "<{}_nonce>".format(word)
