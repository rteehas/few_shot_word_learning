from datasets import load_from_disk
from torch.utils.data import default_collate
import torch
import random
import functools
import numpy as np


def custom_collate_fn(batch):
    baseline = [{k: v for k, v in b.items() if k not in ['firstSpan', 'secondSpan', "generationTokens"]} for b in batch]
    base_collate = default_collate(baseline)
    if "firstSpan" in batch[0] and "secondSpan" in batch[0]:
        firstSpan = [b["firstSpan"] for b in batch]
        secondSpan = [b["secondSpan"] for b in batch]

        base_collate["firstSpan"] = firstSpan
        base_collate['secondSpan'] = secondSpan
    genToks = [b["generationTokens"] for b in batch]

    base_collate["generationTokens"] = genToks

    return base_collate


def collate_fn_replace_corrupted(batch, dataset):
    """Collate function that allows to replace corrupted examples in the
    dataloader. It expect that the dataloader returns 'None' when that occurs.
    The 'None's in the batch are replaced with another examples sampled randomly.

    Args:
        batch (torch.Tensor): batch from the DataLoader.
        dataset (torch.utils.data.Dataset): dataset which the DataLoader is loading.
            Specify it with functools.partial and pass the resulting partial function that only
            requires 'batch' argument to DataLoader's 'collate_fn' option.

    Returns:
        torch.Tensor: batch with new examples instead of corrupted ones.
    """
    # Idea from https://stackoverflow.com/a/57882783

    original_batch_len = len(batch)
    # Filter out all the Nones (corrupted examples)
    batch = list(filter(lambda x: x is not None, batch))
    filtered_batch_len = len(batch)
    # Num of corrupted examples
    diff = original_batch_len - filtered_batch_len
    if diff > 0:
        # Replace corrupted examples with another examples randomly
        batch.extend([dataset[random.randint(0, len(dataset)-1)] for _ in range(diff)])
        # Recursive call to replace the replacements if they are corrupted
        return collate_fn_replace_corrupted(batch, dataset)
    # Finally, when the whole batch is fine, return it
    return torch.utils.data.dataloader.default_collate(batch)

def make_collate(dataset):
    return functools.partial(collate_fn_replace_corrupted, dataset=dataset)


def make_nonce(num):
    return "<nonce_{}>".format(num)

def load_chimera(trial):
    fname = "chimeras_{}".format(trial)
    return load_from_disk(fname)

def make_sanity_nonce(word):
    return "<{}>".format(word)

def convert_to_base(num: int, base: int, numerals="0123456789abcdefghijklmnopqrstuvwxyz") -> str:
    return ((num == 0) and numerals[0]) or (
        convert_to_base(num // base, base, numerals).lstrip(numerals[0]) + numerals[num % base])


def convert_to_character(number: str, separator: str, invert_number: bool, max_digits: int) -> str:
    if max_digits > 0:
        signal = None
        if number[0] == '-':
            signal = '-'
            number = number[1:]
        number = (max_digits - len(number)) * '0' + number
        if signal:
            number = signal + number
    if invert_number:
        number = number[::-1]
    return separator.join(number)


def get_span(sent, word, tokenizer):
    enc = tokenizer.encode_plus(sent)
    word_id = None
    for i in enc.word_ids():
        if i is not None:
            start, end = enc.word_to_chars(i)
            if word == sent[start: end]:
                word_id = i

    if word_id is not None:
        return np.where(np.array(enc.word_ids()) == word_id)[0].tolist()
    else:
        raise Exception("Word {} not in sentence {}".format(word, sent))