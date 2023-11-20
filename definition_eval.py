import numpy as np
from train_with_llama import *
from torch.optim import adamw
from train_with_llama import *
from transformers import RobertaForMaskedLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, \
    get_linear_schedule_with_warmup, AdamW, DataCollatorForLanguageModeling, AutoConfig


device = "cuda"

definition_prompt = "Given the following examples: {}, give the definition of a new word \"{}\".\n Definition:"

def generate_definitions_emb_gen(model, ex, k, with_prompt):
    examples = np.random.choice(ex['replaced_examples'], size=k, replace=False)
    context = tokenizerMLM(examples.tolist(), truncation=True, padding='longest', return_tensors='pt')
    nonce = "<{}_new>".format(ex['word'].lower())
    if with_prompt:
        prompt = definition_prompt.format("\n".join(examples), nonce)
    else:
        prompt = "{} is defined as".format(nonce)

    inputs = tokenizerTask(prompt, truncation=True, return_tensors='pt', max_length=256).to(device)

    outputs = generate(model, context, inputs['input_ids'], inputs['attention_mask'], 100, temperature=1.0, top_k=None, do_sample=False)
    generated_def = tokenizerTask.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    new_ex = {'definition': ex['definition'],
           'word':ex['word'],
           'generated definition': generated_def}
    return new_ex

@torch.no_grad
def generate_definitions(ex,k):
    examples = np.random.choice(ex['replaced_examples'], size=k, replace=False)
    example_prompt = definition_prompt.format("\n".join(examples), "<nonce>")
    inputs = tokenizer(example_prompt, return_tensors='pt')
    output = model.generate(**inputs.to(device), max_new_tokens=100)
    generated_def = tokenizer.decode(output[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    new_ex = {'definition': ex['definition'],
           'word':ex['word'],
           'generated definition': generated_def}
    return new_ex

@torch.no_grad
def generate_definitions_examples(model, tokenizer, ex, examples, with_prompt):
    if with_prompt:
        example_prompt = definition_prompt.format("\n".join(examples), "<nonce>")
    else:
        example_prompt = "The word <nonce> means"
    inputs = tokenizer(example_prompt, return_tensors='pt')
    output = model.generate(**inputs.to(device), max_new_tokens=100)
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
    inputs = tokenizerTask(examples, truncation=True, padding='longest', return_tensors='pt').to(model.device)
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

def run_baseline(def_task):
    max_num_steps = 8
    fname_format = "definition_task_outputs/baseline_generations_lr_{}"
    for lr in [3e-4, 1e-3, 3e-3]:
        all_outputs = []
        for k in range(1,4):
            for ex in def_task:
                tokenizerTask = LlamaTokenizer.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                               legacy=True,
                                                               use_fast=False)
                secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf",
                                                            low_cpu_mem_usage=True).to(device)
                tokenizerTask.add_tokens(["<nonce>"])
                secondLM.resize_token_embeddings(len(tokenizerTask))
                step_outputs = generate_definitions_examples(secondLM, tokenizerTask,ex, k, max_num_steps, lr)

                all_outputs += step_outputs

        save_dir = fname_format.format(lr)
        keys = all_outputs[0].keys()
        data_dict = {}
        for key in keys:
            data_dict[key] = [output_ex[key] for output_ex in all_outputs]

        Dataset.from_dict(data_dict).save_to_disk(fname_format)


def run_emb_gen(def_task, args):
    pass

if __name__ == "__main__":
    def_task = load_from_disk("initial_wordnet_def_task")
    def_task = def_task.map(replace_for_llama_baseline)
    run_baseline(def_task)




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
