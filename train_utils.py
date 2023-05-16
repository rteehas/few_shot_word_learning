import torch
from torch import nn

def get_grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def save(model, opt, scheduler, model_name_chkpt):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        
        "/scratch/rst306/few_shot_word_learning/checkpoints/{}.pth".format(model_name_chkpt))


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


def get_nonce_loss(batch, out, vocab_size, device):
    b_task, k_task, l_task = batch["task_inputs"]["input_ids"].shape

    nonceTask = batch['nonceTask'].to(device)
    task_labels = batch["task_labels"].to(device).reshape((b_task * k_task, l_task))

    token_loss = get_per_token_loss(task_labels, out.logits, nonceTask, vocab_size)
    if token_loss.numel() > 0:
        return token_loss.mean()
    else:
        return None


def get_model_name_checkpoint(epoch, dataset_name, model):
    return "{}_{}_epoch={}".format(dataset_name, epoch, model.model_name)