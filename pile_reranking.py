import torch
from datasets import load_from_disk
from scipy.spatial.distance import cosine
import numpy as np
import re
from functools import partial

from transformers import AutoTokenizer, RobertaForMaskedLM

sent_embeds = torch.load("medium_sentence_embeds_train", map_location="cpu")
text_embeds = torch.load("medium_text_embeds_train", map_location="cpu")
ctr = 0


def revert_sentences(ex):
    sentences = ex['sentences']
    nonce = ex['word']
    word = get_word(ex['word'])
    sentences = [s.replace(nonce, word) for s in sentences]
    ex['reverted_sentences'] = sentences
    return ex


def get_word_idx(sent, word):
    try:
        return sent.split(" ").index(word)
    except ValueError:
        split = sent.split(" ")
        for i, x in enumerate(split):
            if word in x:
                return i


def get_locs(sent, idx, tokenizer):
    encoded = tokenizer.encode_plus(sent, return_tensors="pt", truncation=True)
    token_ids_word = np.where(np.array(encoded.word_ids()) == idx)
    return token_ids_word


def get_word(text):
    word_matches = re.findall(r'<(\w+)_new>', text)
    return word_matches[0]


def get_word_embed_in_sentence(sentence, word, tokenizer, sent_embed):
    idx = get_word_idx(sentence, word)
    locs = get_locs(sentence, idx, tokenizer)

    return sent_embed[locs].mean(dim=0)


def get_scores(text, sentences, word, tokenizer):
    text_embed = get_word_embed_in_sentence(text, word, tokenizer, text_embeds[text]).numpy()
    sentence_embeds = []
    for s in sentences:
        emb = get_word_embed_in_sentence(s, word, tokenizer, sent_embeds[s]).numpy()
        sentence_embeds.append(emb)
    similarities = [1 - cosine(text_embed, s) for s in sentence_embeds]
    return similarities


def score_sentences(ex, tokenizer=None):
    global ctr
    ctr += 1
    if ctr % 100 == 0:
        print(ctr)
    base_sentences = ex['sentences']
    sentences = ex['reverted_sentences']
    word = get_word(ex['word'])
    text = ex['base text']
    scores = get_scores(text, sentences, word, tokenizer)
    ex['scores'] = scores
    return ex


def rank_by_score(ex):
    scores = ex['scores']
    base_sentences = ex['sentences']
    zipped = zip(base_sentences, scores)
    sorted_sentences = sorted(zipped, key=lambda x: x[1], reverse=True)
    ex['top_ranked_sentences'] = [x[0] for x in sorted_sentences[:7]]
    return ex


def filter_by_score(ex):
    scores = ex['scores']
    filtered_scores = [s for s in scores if s >= 0.85]
    if len(filtered_scores) < 7:
        return False
    else:
        return True


def main():
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base", use_fast=True)
    partial_scores = partial(score_sentences, tokenizer=tokenizer)
    print("loading data")
    data = load_from_disk("pile_medium_regression_v3")
    print("filtering for second half")
    data = data['train'].filter(lambda ex: len(ex['sentences']) > 100)
    print("reverting sentences")
    data = data.map(revert_sentences)
    print("scoring")
    scored = data.map(partial_scores)
    print("filtering")
    scored = scored.filter(filter_by_score)
    print("ranking")
    scored = scored.map(rank_by_score)
    print("saving")
    scored.save_to_disk("pile_second_half_ranked")


if __name__ == "__main__":
    main()
