import torch
import numpy as np

def get_tgt_mask(size):
    # Generates a squeare matrix where the each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

    # EX for size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]

    return mask


def get_word_idx(sent, word):
    return sent.split(" ").index(word)


def combine_layers(hidden_states, layers):
    return torch.stack([hidden_states[i] for i in layers]).sum(0).squeeze()


def extract_word_embeddings(hidden_states, layers, word_index, encoded):
    tp = torch.stack([hidden_states[i] for i in layers]).sum(0).squeeze()
    token_ids_word = np.where(np.array(encoded.word_ids()) == word_index)
    out = tp[token_ids_word]
    return out.mean(dim=0)
