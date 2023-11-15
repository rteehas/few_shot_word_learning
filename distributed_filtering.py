from typing import Callable
import shutil
import re
import datasets
import torch
from datasets import load_from_disk
from transformers import RobertaForMaskedLM, AutoTokenizer
import numpy as np
from torch.nn import CosineSimilarity
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import time

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

datasets.disable_caching()
cache_path = "/home/.cache"

def dataset_map_multi_worker(
    dataset: datasets.Dataset, map_fn: Callable, *args, **kwargs
) -> datasets.Dataset:
    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    except (RuntimeError, ValueError):
        return dataset.map(map_fn, *args, **kwargs)
    ds_shard_filepaths = [
        os.path.join(cache_path, f"{dataset._fingerprint}_subshard_{w}.cache")
        for w in range(0, world_size)
    ]
    print(f"\tworker {rank} saving sub-shard to {ds_shard_filepaths[rank]}")
    ds_shard = dataset.shard(
        num_shards=world_size,
        index=rank,
        contiguous=True,
    )
    ds_shard = ds_shard.map(map_fn, *args, **kwargs)
    ds_shard.save_to_disk(ds_shard_filepaths[rank])
    print("rank", rank, "saving:", ds_shard_filepaths[rank])
    torch.distributed.barrier()
    full_dataset = datasets.concatenate_datasets(
        [datasets.load_from_disk(p) for p in ds_shard_filepaths]
    )
    torch.distributed.barrier()
    print("rank", rank, "deleting:", ds_shard_filepaths[rank])
    shutil.rmtree(ds_shard_filepaths[rank])
    return full_dataset



def main(rank, world_size):
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

    def get_emb(hidden, locs, layers, idx):
        output = torch.stack([hidden[i][idx, :, :] for i in layers]).sum(0).squeeze()

        word_tokens_output = output[locs]
        return word_tokens_output.mean(dim=0)

    def get_word(text):
        word_matches = re.findall(r'<(\w+)_new>', text)
        return word_matches[0]

    def revert_sentences(ex):
        sentences = ex['sentences']
        nonce = ex['word']
        word = get_word(ex['word'])
        sentences = [s.replace(nonce, word) for s in sentences]
        ex['reverted_sentences'] = sentences
        return ex

    @torch.no_grad
    def get_word_emb(encoded, token_ids_word, model, layers):
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True)

        states = output.hidden_states

        output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

        word_tokens_output = output[token_ids_word]

        return word_tokens_output.mean(dim=0)

    def get_word_embed_in_sentence(sentence, word):
        layers = [-1]
        idx = get_word_idx(sentence, word)
        locs = get_locs(sentence, idx, tokenizer)
        enc = tokenizer(sentence, truncation=True, return_tensors='pt')

        emb = get_word_emb(enc.to(model.device), locs, model, layers)
        return emb

    def get_scores(text, sentences, word):
        text_embed = get_word_embed_in_sentence(text, word)
        sent_embeds = []
        for s in sentences:
            emb = get_word_embed_in_sentence(s, word)
            sent_embeds.append(emb)
        similarities = [cos(text_embed, s) for s in sent_embeds]
        return similarities

    def score_sentences(ex):
        global ctr
        ctr += 1
        if ctr % 100 == 0:
            print(ctr)
        base_sentences = ex['sentences']
        sentences = ex['reverted_sentences']
        word = get_word(ex['word'])
        text = ex['base text']
        scores = get_scores(text, sentences, word)
        ex['scores'] = [score.item() for score in scores]
        return ex
    ctr = 0
    setup(rank, world_size)
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained("distillroberta-base", use_fast=True)
    model = RobertaForMaskedLM.from_pretrained("distillroberta-base", low_cpu_mem_usage=True)
    ddp_model = DDP(model, device_ids=[rank])
    data = load_from_disk("pile_medium_regression_v3")
    datasets.disable_caching()
    cache_path = "$SCRATCH/.cache"
    data = data.map(revert_sentences)
    cos = CosineSimilarity(dim=0)

    output = dataset_map_multi_worker(data, score_sentences)


if __name__ == "__main__":
    world_size = 2
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)