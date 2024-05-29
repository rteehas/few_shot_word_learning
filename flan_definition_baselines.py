import csv
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from argparse import ArgumentParser
import tqdm
from os import path
import json
from datasets import load_from_disk
import numpy as np

def load_data(path_to_data, split="test"):
    if "oxford" not in path_to_data and "wordnet" not in path_to_data:
        if split:
            datafile = path.join(path_to_data, f"{split}.complete.tsv.gz")
        else:
            datafile = path.join(path_to_data, f"complete.tsv.gz")
        df = pd.read_csv(datafile, delimiter="\t", header=0, quoting=csv.QUOTE_NONE,
                         encoding="utf-8", on_bad_lines="warn")
        df["Context"] = df.example
        df["Targets"] = [w.split("%")[0] for w in df.word]
        try:
            df["Definition"] = df.gloss
        except AttributeError:
            print("No definitions found in the input file")
            # df["Definition"] = df.example
    else:
        datafile = path.join(path_to_data, split + ".eg.gz")
        datafile_defs = path.join(path_to_data, split + ".txt.gz")
        df = pd.read_csv(datafile, delimiter="\t", quoting=csv.QUOTE_NONE,
                         encoding="utf-8", on_bad_lines="warn")
        df_defs = pd.read_csv(datafile_defs, delimiter="\t", quoting=csv.QUOTE_NONE,
                         encoding="utf-8", on_bad_lines="warn")
        df_defs.columns = ["Sense", "Ignore1", "Ignore2", "Definition", "Ignore3", "Ignore4"]
        df.columns = ["Sense", "Context"]
        df["Targets"] = [w.split("%")[0] for w in df.Sense]
        df["Definition"] = df_defs.Definition
    if "wordnet" in path_to_data:
        df["POS"] = [w.split("%")[1].split(".")[2] for w in df.Sense]
    contexts = [ctxt.replace("<TRG>", targetword).strip() for ctxt, targetword
                in zip(df.Context, df.Targets)]
    df["Real_Contexts"] = contexts
    return df

def define(in_prompts, lm, cur_tokenizer, arguments, targets, filter_target=False, num_beams=1,
        num_beam_groups=1, sampling=False, temperature=1.0, repetition_penalty=1.0):
    print(f"Tokenizing with max length {arguments.maxl}...")
    inputs = cur_tokenizer(
        in_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=arguments.maxl,
    )
    print("Tokenizing finished.")

    target_ids = cur_tokenizer(targets, add_special_tokens=False).input_ids
    target_ids = torch.tensor([el[-1] for el in target_ids])

    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
        target_ids = target_ids.to("cuda")

    test_dataset = torch.utils.data.TensorDataset(inputs["input_ids"], inputs["attention_mask"],
                                                  target_ids)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=arguments.bsize, shuffle=False)
    print(f"Generating definitions with batch size {arguments.bsize}...")
    gen_args = dict(do_sample=sampling, num_beams=num_beams, num_beam_groups=num_beam_groups,
            temperature=temperature, repetition_penalty=repetition_penalty)
    if num_beam_groups > 1:
        gen_args["diversity_penalty"] = 0.5
    definitions = []
    for inp, att, targetwords in tqdm.tqdm(test_iter):
        if filter_target:
            bad = [[el] for el in targetwords.tolist()]
            outputs = lm.generate(input_ids=inp, attention_mask=att, max_new_tokens=60,
                                  bad_words_ids=bad, **gen_args)
        else:
            outputs = lm.generate(input_ids=inp, attention_mask=att, max_new_tokens=60,
                                  **gen_args)
        predictions = cur_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        definitions += predictions
    print(f"Generating definitions finished")
    return definitions

def run_oxford(flan_model, task_instructions):
    device = "cuda"
    data = load_data("data/oxford")
    tokenizer = AutoTokenizer.from_pretrained("ltg/flan-t5-definition-en-{}".format(flan_model))
    model = AutoModelForSeq2SeqLM.from_pretrained("ltg/flan-t5-definition-en-{}".format(flan_model))
    model = model.to(device)

    for task_prefix in task_instructions:
        print(f"Generating with the task instruction {task_prefix}...", flush=True)
        identifier = "_".join(task_prefix).lower().replace(" ", "_")
        input_sentences = []
        for target, context in zip(data.Targets, data.Real_Contexts):
            if task_prefix[1] == "pre":
                prompt = " ".join([task_prefix[0].replace("<TRG>", target), context])
            else:
                prompt = " ".join([context, task_prefix[0].replace("<TRG>", target)])

            input_sentences.append(prompt)
        answers = define(input_sentences, model, tokenizer, args, data.Targets.tolist(),
                            filter_target=True, sampling=False,
                            repetition_penalty=1.0, num_beams=1,
                            num_beam_groups=1)

        data["Generated_Definition"] = answers

    merged_data = pd.read_csv("merged_oxford_test_set.csv")
    tmp = data.copy()
    tmp.columns = ['word', 'Context', 'Targets', 'Definition', 'Real_Contexts',
       'Generated_Definition']

    pd.merge(merged_data, tmp, on="word", how="inner").drop(
        ["definition", "sentence", "replaced_examples", "tag", "source", "Unnamed: 0"],
        axis=1,
    ).to_csv("oxford_task_outputs/flan_{}".format(flan_model))

    return data

def run_def_task(flan_model, task_instructions, setting):
    device = "cuda"
    def_task = load_from_disk("def_task_954")
    tokenizer = AutoTokenizer.from_pretrained("ltg/flan-t5-definition-en-{}".format(flan_model))
    model = AutoModelForSeq2SeqLM.from_pretrained("ltg/flan-t5-definition-en-{}".format(flan_model))
    model = model.to(device)
    placeholder = "bax"
    for trial in range(5):
        for task_prefix in task_instructions:
            print(f"Generating with the task instruction {task_prefix}...")
            identifier = "_".join(task_prefix).lower().replace(" ", "_")
            input_sentences = []
            for i, ex in enumerate(def_task):
                if setting == "original_word":
                    context = np.random.choice(ex['gpt_examples'], size=1, replace=False)[0]
                    target = ex['word']
                    
                elif setting == "new_token":
                    context = np.random.choice(ex['replaced_examples'], size=1, replace=False)[0]
                    target = "<nonce>"
                    
                elif setting == "placeholder_word":
                    context = np.random.choice(ex['replaced_examples'], size=1, replace=False)[0]
                    context = context.replace("<nonce>", placeholder)
                    target = placeholder
                else:
                    raise NotImplementedError
                    
                if task_prefix[1] == "pre":
                    prompt = " ".join([task_prefix[0].replace("<TRG>", target), context])
                else:
                    prompt = " ".join([context, task_prefix[0].replace("<TRG>", target)])
        
                input_sentences.append(prompt)
                # if i == 3:
                #     break
                
            answers = define(input_sentences, model, tokenizer, args, def_task['word'],
                                filter_target=True, sampling=False,
                                repetition_penalty=1.0, num_beams=1,
                                num_beam_groups=1)
            
            # data["Generated_Definition"] = answers
            # id = uuid.uuid4()
            def_task.add_column("generated definition", answers)
            def_task.add_column("prompt", input_sentences)
            def_task.save_to_disk("definition_task_outputs/flan_{}_{}_definitions_{}".format(flan_model, setting, trial))


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--maxl", type=int, default=256)
    parser.add_argument("--bsize", type=int, default=4)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--setting", type=str, default="original_word")
    parser.add_argument("--model", type=str, default="base")

    return parser

if __name__ == "__main__":
    args = get_arguments().parse_args()
    print(args, flush=True)

    prompts = [
        ["", "post"],  # 0
        ["Give the definition of <TRG>:", "pre"],  # 1
        ["Define <TRG>:", "pre"],  # 2
        ["Define the word <TRG>:", "pre"],  # 3
        ["What is the definition of <TRG>?", "pre"],  # 4
        ["Give the definition of <TRG>", "post"],  # 5
        ["Define <TRG>", "post"],  # 6
        ["Define the word <TRG>", "post"],  # 7
        ["What is the definition of <TRG>?", "post"], #  8
        ["Quelle est la définition de <TRG>?", "post"],  # 9
        ["Что такое <TRG>?", "post"],  # 10
        ["Hva betyr <TRG>?", "post"],  # 11
        ["Was ist die Definition von <TRG>?", "post"],  # 12
    ]

    task_instructions = [prompts[8]]

    setting = args.setting
    if args.dataset == "oxford":
        results = run_oxford(args.model, task_instructions)
    elif args.dataset == "def_task":
        results = run_def_task(args.model, task_instructions, setting)
    else:
        raise NotImplementedError

    