import torch
import numpy as np


def get_word_idx(sent, word):
    return sent.split(" ").index(word)


def combine_layers(hidden_states, layers):
    return torch.stack([hidden_states[i] for i in layers]).sum(0).squeeze()

def extract_word_embeddings(hidden_states, layers, word_index, encoded):
    tp = torch.stack([hidden_states[i] for i in layers]).sum(0).squeeze()
    token_ids_word = np.where(np.array(encoded.word_ids()) == word_index)
    out = tp[token_ids_word]
    return out.mean(dim=0)
