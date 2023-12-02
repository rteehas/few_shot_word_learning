import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from argparse import ArgumentParser
from datasets import load_from_disk, Dataset, DatasetDict
from accelerate import PartialState
from transformers import pipeline
import re

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    return parser


def generate_data(pipe, prefixes, num_beams, max_new_tokens):

    results = pipe(prefixes, use_cache=True, num_beams=num_beams, max_new_tokens=max_new_tokens)

    return list(zip(prefixes, results))

def get_word(w):
    matches = re.findall(r"<(\w+)_new>",w)
    return matches[0]

def get_prefixes(ex):
    nonce = ex['word']
    word = get_word(nonce)

    prefix = ex['text'].split()[0]
    prefix = prefix + word
    ex['prefix'] = prefix
    return ex

def main():

    args = get_arguments().parse_args()

    data = load_from_disk(args.data_path)
    data = data.map(get_prefixes)

    orig_dset_name = args.data_path.split("/")[-1]
    distributed_state = PartialState()
    model = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                low_cpu_mem_usage=True, device_map=distributed_state.device)
    tokenizer = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", legacy=True,
                                   use_fast=False)


    prefixes = list(set(data['train']['prefix'] + data['test']['prefix']))

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=distributed_state.device)

    with distributed_state.split_between_processes(prefixes) as partial_prefixes:

        outputs = generate_data(pipe, partial_prefixes, 3, 128)

        data_dict = {'prefix': [o[0] for o in outputs], 'text': [o[1][0]['generated_text'] for o in outputs]}

        train_set = Dataset.from_dict(data_dict)

        train_set.save_to_disk("{}_llama_generations_{}".format(orig_dset_name, distributed_state.process_index))


if __name__ == "__main__":
    main()
