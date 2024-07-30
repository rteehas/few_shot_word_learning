import numpy as np
from train_with_llama import *
from torch.optim import adamw
from train_with_llama import *
from transformers import RobertaForMaskedLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, \
    get_linear_schedule_with_warmup, AdamW, DataCollatorForLanguageModeling, AutoConfig
from transformers import PreTrainedTokenizerFast
from copy import deepcopy
from datasets import Dataset
# from run_gre_eval_llama import extract_arguments_from_path
# from w2v_baselines import HiCEBaseline, load_dictionary, make_hice_batch, generate_hice, AdditiveBaseline, \
#     generate_additive
import uuid
import pandas as pd
from datetime import datetime
import evaluate
from datasets import load_dataset

# Load the BERTScore metrics
bertscore_metric = evaluate.load('bertscore')

def calculate_bertscore(reference, candidate):
    # Specify the device as 'cuda' to use the GPU
    results = bertscore_metric.compute(predictions=[candidate], references=[reference], lang='en', device='cuda', model_type='microsoft/deberta-large-mnli')
    return results['f1'][0]  # results['f1'] is a list, get the first element for the current pair

# Check the number of available GPUs
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

if num_gpus >= 2:
    device_0 = torch.device('cuda:0')
    device_1 = torch.device('cuda:1')
else:
    device_0 = torch.device('cuda:0')
    device_1 = torch.device('cuda:0')  # Fallback to the same device if only one GPU is available

definition_prompt = "Given the following examples: {}, the word \"{}\" is defined as"
basic_definition_prompt = "The word \"{}\" is defined as".format("<nonce>")

example_prompt = "Given the following examples for a new word \"<nonce>\": {}, another example sentence for the word \"{}\" is:"
basic_example_prompt = "Another example sentence for the word \"{}\" is:".format("<nonce>")

@torch.no_grad()
def load_model_and_tokenizer(model_name, device):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer


@torch.no_grad()
def generate_text_completion(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
    tokenizer.pad_token = tokenizer.eos_token
    # Encode the prompt and create attention mask
    inputs = tokenizer.encode_plus(prompt, return_tensors='pt', padding=False, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)
    # Generate text completion with attention mask and pad token id
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    completion = tokenizer.decode(output[0], skip_special_tokens=True)
    # Remove the prompt from the completion
    completion = completion[len(prompt):]
    # Take the first sentence from the completion
    first_sentence = completion.split('.')[0]

    return first_sentence

@torch.no_grad()
def generate_definition(model, examples, tokenizerMLM, tokenizerTask, with_prompt):
    context = tokenizerMLM(examples, truncation=True, padding='longest', return_tensors='pt')
    nonce = "<nonce>"
    if with_prompt:
        prompt = definition_prompt.format("\n".join(examples), nonce)
    else:
        prompt = "The word \"{}\" is defined as".format(nonce)

    inputs = tokenizerTask(prompt, truncation=True, return_tensors='pt', max_length=256).to(model.device)

    # Generate definition using the full prompt
    outputs = generate(model, context, inputs['input_ids'], inputs['attention_mask'], 30, mask_new_tokens=False)
    generated_def = tokenizerTask.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    return generated_def

    

@torch.no_grad()
def generate_examples_emb_gen(model, ex, tokenizerMLM, tokenizerTask, llama_model, llama_tokenizer, with_prompt=True, temperature=1):
    examples = ex['examples']
    print(examples)
    context = tokenizerMLM(examples, truncation=True, padding='longest', return_tensors='pt')
    nonce = "<nonce>"
    if with_prompt:
        prompt = example_prompt.format("\n".join(examples), nonce)
    else:
        prompt = basic_example_prompt 

    inputs = tokenizerTask(prompt, truncation=True, return_tensors='pt', max_length=256).to(model.device)
    generated_examples = []
    
    # start another loop
    for _ in range(4):
        generated_exs = []
        try:
            context = tokenizerMLM(examples, truncation=True, padding='longest', return_tensors='pt')
        except Exception as e:
            print("An error occurred during tokenization. Here are the examples:")
            print(examples)
            raise e  # Re-throw the error to handle it further up the call stack or halt the program
        for _ in range(3):  # Loop for 3 generations
            # Generate example using LLaMA model and tokenizer
            gen_ex = generate_text_completion(llama_model, llama_tokenizer, prompt,  temperature=temperature)
            # remove the prompt from the generated example
            examples.append(gen_ex)
            gen_def = generate_definition(model, examples, tokenizerMLM, tokenizerTask, with_prompt=False)
            # Remove the generated example from the list
            examples.pop()
            generated_exs.append({'example': gen_ex, 'definition': gen_def})

        
        # Calculate BERTScore for each generated definition against ex['definition']
        scores_with_exs = [(calculate_bertscore(ex['definition'], gen_ex['definition']), gen_ex) for gen_ex in generated_exs]
        generated_examples.append(generated_exs)
        # Sort based on scores and select the top one
        top_score, top_ex = max(scores_with_exs, key=lambda x: x[0])
        # end loop
        examples.append(top_ex['example'])

    new_ex = {'definition': ex['definition'],
              'word': ex['word'],
              'generated examples': generated_examples,
              'examples': examples,
              'example gen prompt': prompt}
    return new_ex

@torch.no_grad()
def generate_definitions_emb_gen(model, ex, tokenizerMLM, tokenizerTask, with_prompt):
    examples = ex['examples']
    context = tokenizerMLM(examples, truncation=True, padding='longest', return_tensors='pt')
    nonce = "<nonce>"
    if with_prompt:
        prompt = definition_prompt.format("\n".join(examples), nonce)
    else:
        prompt = "The word \"{}\" is defined as".format(nonce)

    inputs = tokenizerTask(prompt, truncation=True, return_tensors='pt', max_length=256).to(model.device)

    # Generate definition using the full prompt
    outputs = generate(model, context, inputs['input_ids'], inputs['attention_mask'], 30, mask_new_tokens=False)
    generated_def = tokenizerTask.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

    examples = [examples[0]]
    context = tokenizerMLM(examples, truncation=True, padding='longest', return_tensors='pt')
    outputs_original = generate(model, context, inputs['input_ids'], inputs['attention_mask'], 30, mask_new_tokens=False)
    original_generated_def = tokenizerTask.decode(outputs_original[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

    new_ex = ex.copy()
    new_ex['self play generated definition'] = generated_def
    new_ex['original generated definition'] = original_generated_def  # Add the original generated definition
    if with_prompt:
        new_ex['definition gen prompt'] = definition_prompt
    else:
        new_ex['definition gen prompt'] = "The word \"{}\" is defined as"
    return new_ex

def run_emb_gen(def_task, path, temperature=1):
    # config_args = extract_arguments_from_path(args.path)
    # Generate a timestamp in a specific format, e.g., YYYYMMDDHHMMSS
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    # Generate a random UUID
    random_uuid = uuid.uuid4()
    # Combine the timestamp with the UUID to ensure chronological ordering
    id = f"{timestamp}-{random_uuid}"
    fname_format = "/scratch/jl16973/few_shot_word_learning/definition_task_outputs/self-play/temp_{}_emb_gen_generations_masked_new_token_new_data_new_model_{}".format(temperature, id)
    tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
    tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
    nonces = list(tokenizerTask.get_added_vocab().keys())
    firstLM = RobertaForMaskedLM.from_pretrained("roberta-large", low_cpu_mem_usage=True)
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", low_cpu_mem_usage=True)
    memory_config = AggregatorConfig()

    mask_token_id = tokenizerMLM.mask_token_id
    layers=[-1]
    model = MorphMemoryModelLLAMA(firstLM, secondLM, len(nonces), layers, mask_token_id, memory_config, 1, None).to(device_0)
    model.emb_gen.load_state_dict(torch.load(path + "/pytorch_model.bin"))
    model.device = device_0
    model.firstLM.eval()
    model.secondLM.eval()
    model.eval()

    model_path_llama3 = '/vast/work/public/ml-datasets/llama-3/Meta-Llama-3-8B-Instruct-hf'
    model_llama3, tokenizer_llama3 = load_model_and_tokenizer(model_path_llama3, device_1)

    all_outputs = []
    # Assuming the custom prompt parameter is added to the function as discussed
    counter = 0
    # Assuming the custom prompt parameter is added to the function as discussed
    batch_size = 20
    batch_outputs = []  # Temporary list to store outputs for every batch
    for i, ex in enumerate(def_task):
        if i >= 40:  # Break the loop after processing 20 examples
            break
        print(f"Processing example {i + 1}...")
        print(ex)
        # Generate examples with prompt but the definition without prompt
        # ex = generate_examples_emb_gen(model, ex, tokenizerMLM, tokenizerTask, with_prompt=True, temperature=temperature)
        ex = generate_examples_emb_gen(model, ex, tokenizerMLM, tokenizerTask, model_llama3, tokenizer_llama3, with_prompt=True, temperature=temperature)
        # Generate output with prompt
        step_output_with_prompt = generate_definitions_emb_gen(model, ex, tokenizerMLM, tokenizerTask, with_prompt=True)
        # Generate output without prompt
        step_output_without_prompt = generate_definitions_emb_gen(model, ex, tokenizerMLM, tokenizerTask, with_prompt=False)
        batch_outputs.extend([step_output_with_prompt, step_output_without_prompt])
        
    # After breaking out of the loop, save the processed examples
    print("Saving processed examples...")
    df = pd.DataFrame(batch_outputs)
    df.to_csv(fname_format + '.csv', mode='w', header=True, index=False)  # Save all at once, assuming fname_format is defined
    print('Saved processed examples to:', fname_format + '.csv')
    return batch_outputs  # Assuming you want to return the processed outputs

def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--path", type=str)
    parser.add_argument("--model", type=str)
    return parser


if __name__ == "__main__":
    args = get_arguments().parse_args()
    def_task = load_from_disk("subset.arrow")
    path = "/scratch/jl16973/college_pretrained_model/checkpoint_7_28000"
    temp = 0.8
    run_emb_gen(def_task, path, temperature=temp)
