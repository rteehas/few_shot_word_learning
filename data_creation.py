from datasets import load_from_disk, Dataset, DatasetDict, load_dataset
import torch
from nltk.tokenize import word_tokenize, sent_tokenize
import string
from collections import defaultdict, Counter
from functools import partial
from tqdm import tqdm

included = ["Pile-CC", "Books3", "BooksCorpus2"]

def check(ex):
    pile_set = ex['meta']['pile_set_name']
    if pile_set in included:
        return True
    else:
        return False

def get_words_within_range(lb, ub, ctr):
    return {k: v for (k,v) in ctr.items() if v >= lb and v <= ub}

def get_sentences_with_word(word, text):
    sentences = sent_tokenize(text)

    final_sentences = []
    for s in sentences:
        if word in s.lower():
            final_sentences.append(s)
    return final_sentences

def get_words_and_examples(ex, word_set, examples_dict):
    sentences = sent_tokenize(ex['text'].lower())
    for s in sentences:
        sent_words = word_tokenize(s)
        sent_words = list(set([x for x in sent_words if all(l in string.ascii_lowercase for l in x)]))
        word_set.update(sent_words)
    words = word_tokenize(ex['text'].lower())
    words = list(set([x for x in words if all(l in string.ascii_lowercase for l in x)]))
    # word_set.update(words)
    for word in words:
        examples_dict[word].append(ex['text'])
    return ex

def get_examples_and_counts(dataset, max_steps):
    count = 0
    word_set = Counter()
    examples_dict = defaultdict(list)
    with tqdm(total=max_steps) as pbar:
        for example in dataset:
            if count > max_steps:
                break
            if check(example):
                get_words_and_examples(example, word_set, examples_dict)
                count += 1
                pbar.update(1)

    return word_set, examples_dict

