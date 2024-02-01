import numpy as np
from train_with_llama import *
from torch.optim import adamw
from train_with_llama import *
from transformers import RobertaForMaskedLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, \
    get_linear_schedule_with_warmup, AdamW, DataCollatorForLanguageModeling, AutoConfig
from copy import deepcopy
from datasets import Dataset
from run_gre_eval_llama import extract_arguments_from_path
from w2v_baselines import HiCEBaseline, load_dictionary, make_hice_batch, generate_hice, AdditiveBaseline, \
    generate_additive
import uuid

device = "cuda"

definition_prompt = "Given the following examples: {}, the word \"{}\" is defined as"

@torch.no_grad
def generate_definitions_emb_gen(model, ex, k, tokenizerMLM, tokenizerTask, with_prompt):
    examples = np.random.choice(ex['replaced_examples'], size=k, replace=False)
    context = tokenizerMLM(examples.tolist(), truncation=True, padding='longest', return_tensors='pt')
    # nonce = "<{}_new>".format(ex['word'].lower())
    nonce = "<nonce>"
    if with_prompt:
        prompt = definition_prompt.format("\n".join(examples), nonce)
    else:
        prompt = "The word \"{}\" is defined as".format(nonce)

    inputs = tokenizerTask(prompt, truncation=True, return_tensors='pt', max_length=256).to(device)

    outputs = generate(model, context, inputs['input_ids'], inputs['attention_mask'], 30, mask_new_tokens=True)
    # print(outputs)
    generated_def = tokenizerTask.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    # print(ex['word'], generated_def)
    new_ex = {'definition': ex['definition'],
           'word':ex['word'],
           'generated definition': generated_def,
            'examples': examples.tolist(),
            'prompt': prompt}
    return new_ex

@torch.no_grad
def generate_definitions(ex,k):
    examples = np.random.choice(ex['replaced_examples'], size=k, replace=False)
    example_prompt = definition_prompt.format("\n".join(examples), "<nonce>")
    inputs = tokenizer(example_prompt, return_tensors='pt')
    output = model.generate(**inputs.to(device), max_new_tokens=30)
    generated_def = tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    new_ex = {'definition': ex['definition'],
           'word':ex['word'],
           'generated definition': generated_def}
    return new_ex

@torch.no_grad
def generate_definitions_examples(model, tokenizer, ex, examples, with_prompt):
    nonce = "<nonce>"
    if with_prompt:
        example_prompt = definition_prompt.format("\n".join(examples), "<nonce>")
    else:
        example_prompt = "The word \"{}\" is defined as".format(nonce)
    inputs = tokenizer(example_prompt, return_tensors='pt')
    output = model.generate(**inputs.to(device), max_new_tokens=30, use_cache=True)
    generated_def = tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    new_ex = {'definition': ex['definition'],
              'word': ex['word'],
              'generated definition': generated_def,
              'prompt': example_prompt}
    return new_ex

@torch.no_grad
def generate_definitions_examples_hice(model, tokenizer, ex, examples, dictionary, with_prompt):
    nonce = "<nonce>"
    if with_prompt:
        example_prompt = definition_prompt.format("\n".join(examples), "<nonce>")
    else:
        example_prompt = "The word \"{}\" is defined as".format(nonce)
    b = make_hice_batch(examples, ex['wordnet_word'], dictionary, 24, 0)
    context = b['contexts'].to(model.device)
    vocab = b['character'].to(model.device)
    # print(context)
    # print(vocab)
    inputs = tokenizer(example_prompt, return_tensors='pt').to(model.device)
    output = generate_hice(model, context, vocab, inputs['input_ids'], inputs['attention_mask'], 30)
    generated_def = tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    new_ex = {'definition': ex['definition'],
              'word': ex['word'],
              'generated definition': generated_def,
              'prompt': example_prompt}
    return new_ex

@torch.no_grad
def generate_definitions_examples_additive(model, tokenizer, ex, examples, dictionary, with_prompt):
    if with_prompt:
        example_prompt = definition_prompt.format("\n".join(examples), "<nonce>")
    else:
        example_prompt = "The word <nonce> means"
    b = make_hice_batch(examples, ex['wordnet_word'], dictionary, 24, 0)
    context = b['contexts'].to(model.device)
    # vocab = b['character'].to(model.device)
    inputs = tokenizer(example_prompt, return_tensors='pt').to(model.device)
    output = generate_additive(model, context, inputs['input_ids'], inputs['attention_mask'], 30)
    generated_def = tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    new_ex = {'definition': ex['definition'],
              'word': ex['word'],
              'generated definition': generated_def,
              'prompt': example_prompt}
    return new_ex



def replace_for_llama_baseline(ex):
    examples = ex['replaced_examples']
    nonce_format = "<{}_new>"
    nonce = nonce_format.format(ex['word'].lower())
    ex['replaced_examples'] = [s.replace(nonce, "<nonce>") for s in examples]
    return ex

def gradient_descent_tuning(model, tokenizerTask, ex, k, num_steps, lr):
    new_tok_index = tokenizerTask.convert_tokens_to_ids("<nonce>")
    zero_grad_indices = torch.arange(0, len(tokenizerTask)) != new_tok_index
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_input_embeddings().parameters():
        param.requires_grad = True
    for param in model.get_output_embeddings().parameters():
        param.requires_grad = True

    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=lr)
    examples = np.random.choice(ex['replaced_examples'], size=k, replace=False).tolist()
    inputs = tokenizerTask(examples, padding='longest', return_tensors='pt').to(model.device)
    model.train()
    all_step_outputs = []
    for step in range(num_steps):

        model.zero_grad()
        opt.zero_grad()
        output = model(input_ids=inputs['input_ids'],
                       attention_mask=inputs['attention_mask'],
                       labels=inputs['input_ids'].clone())
        loss = output.loss
        loss.backward()

        opt.step()
        model.get_input_embeddings().weight.grad[zero_grad_indices] = 0.
        model.get_output_embeddings().weight.grad[zero_grad_indices] = 0.

        new_ex_with_prompt = generate_definitions_examples(model, tokenizerTask, ex, examples, with_prompt=True)
        new_ex_without_prompt = generate_definitions_examples(model, tokenizerTask, ex, examples, with_prompt=False)
        for new_ex in [new_ex_with_prompt, new_ex_without_prompt]:
            output_dict = {'step': step + 1,
                           'lr': lr,
                           'num_examples': k}
            for key in new_ex:
                output_dict[key] = new_ex[key]

            all_step_outputs.append(output_dict)

    return all_step_outputs

def run_baseline(def_task, lr):
    max_num_steps = 2
    id = uuid.uuid4()
    fname_format = "definition_task_outputs/baseline_generations_lr_{}_{}"
    print(lr)
    all_outputs = []
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                low_cpu_mem_usage=True).to(device)


    tokenizerTask = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                   legacy=True,
                                                   use_fast=False)
    tokenizerTask.pad_token = tokenizerTask.unk_token
    tokenizerTask.add_tokens(["<nonce>"])
    secondLM.resize_token_embeddings(len(tokenizerTask))
    orig_input_embeds = deepcopy(secondLM.get_input_embeddings())
    orig_output_embeds = deepcopy(secondLM.get_output_embeddings())
    for k in range(1,4):
        print("k", k)
        for ex in def_task:

            step_outputs = gradient_descent_tuning(secondLM, tokenizerTask,ex, k, max_num_steps, lr)

            secondLM.set_input_embeddings(orig_input_embeds)
            secondLM.set_output_embeddings(orig_output_embeds)
            all_outputs += step_outputs

    save_dir = fname_format.format(lr, id)
    keys = all_outputs[0].keys()
    data_dict = {}
    for key in keys:
        data_dict[key] = [output_ex[key] for output_ex in all_outputs]
    print("Saving...")
    Dataset.from_dict(data_dict).save_to_disk(save_dir)

def run_baseline_no_gd(def_task):
    max_num_steps = 2
    id = uuid.uuid4()
    fname_format = "definition_task_outputs/baseline_generations_no_lr_{}".format(id)
    # print(lr)
    all_outputs = []
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                low_cpu_mem_usage=True).to(device)


    tokenizerTask = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                   legacy=True,
                                                   use_fast=False)
    tokenizerTask.pad_token = tokenizerTask.unk_token
    tokenizerTask.add_tokens(["<nonce>"])
    secondLM.resize_token_embeddings(len(tokenizerTask))
    # orig_input_embeds = deepcopy(secondLM.get_input_embeddings())
    # orig_output_embeds = deepcopy(secondLM.get_output_embeddings())
    for k in range(1,4):
        print("k", k)
        for ex in def_task:
            examples = np.random.choice(ex['replaced_examples'], size=k, replace=False).tolist()
            new_ex_with_prompt = generate_definitions_examples(secondLM, tokenizerTask, ex, examples, with_prompt=True)
            new_ex_without_prompt = generate_definitions_examples(secondLM, tokenizerTask, ex, examples, with_prompt=False)
            # step_outputs = gradient_descent_tuning(secondLM, tokenizerTask,ex, k, max_num_steps, lr)

            # secondLM.set_input_embeddings(orig_input_embeds)
            # secondLM.set_output_embeddings(orig_output_embeds)
            for new_ex in [new_ex_with_prompt, new_ex_without_prompt]:
                output_dict = {'num_examples': k}
                for key in new_ex:
                    output_dict[key] = new_ex[key]

                all_outputs.append(output_dict)

    # save_dir = fname_format.format(lr)
    save_dir= fname_format
    keys = all_outputs[0].keys()
    data_dict = {}
    for key in keys:
        data_dict[key] = [output_ex[key] for output_ex in all_outputs]
    print("Saving...")
    Dataset.from_dict(data_dict).save_to_disk(save_dir)


def run_emb_gen(def_task, path):
    # config_args = extract_arguments_from_path(args.path)
    id = uuid.uuid4()
    fname_format = "definition_task_outputs/emb_gen_generations_masked_new_token_new_data_new_model_{}".format(id)
    tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
    tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
    nonces = list(tokenizerTask.get_added_vocab().keys())
    # tokenizerMLM.add_tokens(nonces)
    # tokenizerTask.add_tokens(nonces)
    firstLM = RobertaForMaskedLM.from_pretrained("roberta-large", low_cpu_mem_usage=True)
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", low_cpu_mem_usage=True)
    # firstLM.resize_token_embeddings(len(tokenizerMLM))
    # secondLM.resize_token_embeddings(len(tokenizerTask))

    # config_args = extract_arguments_from_path(args.path)
    # print(config_args)
    # if config_args['memory'] == "mean":
    memory_config = AggregatorConfig()
    # elif config_args['memory'] == 'cls':
    # memory_config = TransformerCLSConfig(
    #     input_size=firstLM.config.hidden_size,
    #     nhead=2,
    #     num_layers=1
    # )

    mask_token_id = tokenizerMLM.mask_token_id
    # if 'num_feature_layers' in config_args:
    #     layers = [-1 * (x + 1) for x in range(config_args['num_feature_layers'])]
    # else:
    layers=[-1]
    model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), layers, mask_token_id, memory_config, 1, None).to(device)
    model.emb_gen.load_state_dict(torch.load(path + "/pytorch_model.bin"))
    model.device = device
    model.firstLM.eval()
    model.secondLM.eval()

    # new_nonces = list(map(lambda w: "<{}_new>".format(w.lower()), answers))
    # new_nonces = list(set(new_nonces))
    # tokenizerMLM.add_tokens(new_nonces)
    # tokenizerTask.add_tokens(new_nonces)
    # new_token_num = len(list(tokenizerTask.get_added_vocab().keys())) - len(nonces)

    # model.firstLM.resize_token_embeddings(len(tokenizerMLM))
    # model.secondLM.resize_token_embeddings(len(tokenizerTask))
    # model.add_new_tokens(new_token_num)
    model.eval()
    all_outputs = []
    for k in range(1,4):
        print("Examples: " + str(k))
        for ex in def_task:
            # print(ex)
            step_output_with_prompt = generate_definitions_emb_gen(model, ex, k, tokenizerMLM, tokenizerTask, with_prompt=True)
            step_output_without_prompt = generate_definitions_emb_gen(model, ex, k, tokenizerMLM, tokenizerTask, with_prompt=False)
            all_outputs.append(step_output_with_prompt)
            all_outputs.append(step_output_without_prompt)

    # save_dir = fname_format.format(lr)
    keys = all_outputs[0].keys()
    data_dict = {}
    for key in keys:
        data_dict[key] = [output_ex[key] for output_ex in all_outputs]
    print("Saving...")
    Dataset.from_dict(data_dict).save_to_disk(fname_format)
    return all_outputs

def run_hice(def_task):
    # max_num_steps = 2
    device = "cuda"
    id = uuid.uuid4()
    fname_format = "definition_task_outputs/hice_generations_{}".format(id)
    # print(lr)
    all_outputs = []
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                low_cpu_mem_usage=True).to(device)


    tokenizerTask = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                   legacy=True,
                                                   use_fast=False)
    tokenizerTask.pad_token = tokenizerTask.unk_token
    tokenizerTask.add_tokens(["<nonce>"])

    hice_path = "HiCE/save/model.pt"
    input_linear_path = "baseline_mappings/input_linear.pt"
    output_linear_path = "baseline_mappings/output_linear.pt"
    w2v_dir = 'HiCE/data/base_w2v/wiki_all.sent.split.model'
    corpus_dir = "HiCE/data/wikitext-103/"
    hice = HiCEBaseline(hice_path, input_linear_path, output_linear_path, secondLM).to(device)
    hice.device = device
    dictionary = load_dictionary(w2v_dir, corpus_dir, 24)
    # secondLM.resize_token_embeddings(len(tokenizerTask))
    # orig_input_embeds = deepcopy(secondLM.get_input_embeddings())
    # orig_output_embeds = deepcopy(secondLM.get_output_embeddings())
    for k in range(1,4):
        print("k", k)
        for ex in def_task:
            examples = np.random.choice(ex['replaced_examples'], size=k, replace=False).tolist()
            new_ex_with_prompt = generate_definitions_examples_hice(hice, tokenizerTask, ex, examples, dictionary, with_prompt=True)
            new_ex_without_prompt = generate_definitions_examples_hice(hice, tokenizerTask, ex, examples, dictionary, with_prompt=False)
            # step_outputs = gradient_descent_tuning(secondLM, tokenizerTask,ex, k, max_num_steps, lr)

            # secondLM.set_input_embeddings(orig_input_embeds)
            # secondLM.set_output_embeddings(orig_output_embeds)
            for new_ex in [new_ex_with_prompt, new_ex_without_prompt]:
                output_dict = {'num_examples': k}
                for key in new_ex:
                    output_dict[key] = new_ex[key]

                all_outputs.append(output_dict)

    # save_dir = fname_format.format(lr)
    save_dir= fname_format
    keys = all_outputs[0].keys()
    data_dict = {}
    for key in keys:
        data_dict[key] = [output_ex[key] for output_ex in all_outputs]
    print("Saving...")
    Dataset.from_dict(data_dict).save_to_disk(save_dir)

def run_additive(def_task):
    # max_num_steps = 2
    device = "cuda"
    id = uuid.uuid4()
    fname_format = "definition_task_outputs/additive_generations_{}".format(id)
    # print(lr)
    all_outputs = []
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                low_cpu_mem_usage=True).to(device)


    tokenizerTask = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                   legacy=True,
                                                   use_fast=False)
    tokenizerTask.pad_token = tokenizerTask.unk_token
    tokenizerTask.add_tokens(["<nonce>"])

    hice_path = "HiCE/save/model.pt"
    input_linear_path = "baseline_mappings/input_linear.pt"
    output_linear_path = "baseline_mappings/output_linear.pt"
    w2v_dir = 'HiCE/data/base_w2v/wiki_all.sent.split.model'
    corpus_dir = "HiCE/data/wikitext-103/"
    dictionary = load_dictionary(w2v_dir, corpus_dir, 24)
    additive = AdditiveBaseline(dictionary, input_linear_path, output_linear_path, secondLM).to(device)
    additive.device = device

    # secondLM.resize_token_embeddings(len(tokenizerTask))
    # orig_input_embeds = deepcopy(secondLM.get_input_embeddings())
    # orig_output_embeds = deepcopy(secondLM.get_output_embeddings())
    for k in range(1,4):
        print("k", k)
        for ex in def_task:
            examples = np.random.choice(ex['replaced_examples'], size=k, replace=False).tolist()
            new_ex_with_prompt = generate_definitions_examples_additive(additive, tokenizerTask, ex, examples, dictionary, with_prompt=True)
            new_ex_without_prompt = generate_definitions_examples_additive(additive, tokenizerTask, ex, examples, dictionary, with_prompt=False)
            # step_outputs = gradient_descent_tuning(secondLM, tokenizerTask,ex, k, max_num_steps, lr)

            # secondLM.set_input_embeddings(orig_input_embeds)
            # secondLM.set_output_embeddings(orig_output_embeds)
            for new_ex in [new_ex_with_prompt, new_ex_without_prompt]:
                output_dict = {'num_examples': k}
                for key in new_ex:
                    output_dict[key] = new_ex[key]

                all_outputs.append(output_dict)

    # save_dir = fname_format.format(lr)
    save_dir= fname_format
    keys = all_outputs[0].keys()
    data_dict = {}
    for key in keys:
        data_dict[key] = [output_ex[key] for output_ex in all_outputs]
    print("Saving...")
    Dataset.from_dict(data_dict).save_to_disk(save_dir)



def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--path", type=str)
    parser.add_argument("--model", type=str)
    return parser


if __name__ == "__main__":
    args = get_arguments().parse_args()
    def_task = load_from_disk("def_task_954")
    if args.model == "hice":
        run_hice(def_task)
    elif args.model == "additive":
        run_additive(def_task)
    elif args.model == "baseline_gd":
        run_baseline(def_task, args.lr)
    elif args.model == "baseline_no_gd":
        run_baseline_no_gd(def_task)
    elif args.model == "emb_gen":
        path = "model_checkpoints/layers/no_mp/llama/input_and_output/filtered/redone_pile/layernorm/roberta-large/1_layers/last_1/32_batch_size/mean_agg/1_examples/lr_0.001/weight_decay_0.1/with_negatives_and_regression/distillation_weight_0.05_temp_3/output_embedding_cosine/checkpoints/checkpoint_4_16000"
        run_emb_gen(def_task, path)
    else:
        raise NotImplementedError
    # def_task = def_task.map(replace_for_llama_baseline)
    # run_baseline(def_task, args.lr)
    # path = "model_checkpoints/layers/no_mp/llama/input_and_output/filtered/redone_pile/layernorm/roberta-large/1_layers/last_1/32_batch_size/mean_agg/1_examples/lr_0.001/weight_decay_0.1/with_negatives_and_regression/distillation_weight_0.05_temp_3/output_embedding_cosine/checkpoints/checkpoint_2_9000"
    # path="model_checkpoints/layers/no_mp/llama/input_and_output/filtered/pile/layernorm/roberta-large/1_layers/last_1/32_batch_size/mean_agg/1_examples/lr_0.001/weight_decay_0.1/with_negatives_and_regression/distillation_weight_0.05_temp_3/output_embedding_cosine/checkpoints/checkpoint_4_8500"

    # path = "model_checkpoints/layers/no_mp/llama/input_and_output/filtered/pile/layernorm/roberta-large/1_layers/last_1/32_batch_size/cls_agg/4_examples/lr_0.001/weight_decay_0.1/with_negatives_and_regression/distillation_weight_0.05_temp_3/output_embedding_cosine/checkpoints/checkpoint_7_13500"

    # run_baseline_no_gd(def_task)
    # print("running Hice....")
    # run_hice(def_task)
    # print("Running Additive....")
    # run_additive(def_task)


# for k in range(1,4):
#      k_generations = []
#      for ex in baseline_def_task:
#          k_generations.append(generate_definitions(ex, k))
#      predictions = [x['generated definition'] for x in k_generations]
#      references = [x['definition'] for x in k_generations]
#      scores = bertscore.compute(predictions=predictions, references=references, lang='en')
#      print("Results for k = {}".format(k))
#      for key in scores:
#          if "hash" not in key:
#              mean = np.mean(np.array(scores[key]))
#              std = np.std(np.array(scores[key]))
#              print(key, mean, std)
