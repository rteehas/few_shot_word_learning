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
import pickle
from treelib import Tree



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
    first_sentence = re.split(r'[.\n]', completion)[0]
    # Check if the first sentence is empty
    if not first_sentence.strip():
        first_sentence = completion
    # Check if the first sentence is still empty
    if not first_sentence.strip():
        first_sentence = "empty sentence"
    
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


class TreeNode:
    def __init__(self, example, all_examples, definition_with_prompt, definition_without_prompt):
        self.example = example
        self.all_examples = all_examples
        self.definition_with_prompt = definition_with_prompt
        self.definition_without_prompt = definition_without_prompt
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __str__(self, return_tree_object=False):
        tree = Tree()
        self._build_tree_structure(tree)
        if return_tree_object:
            return tree
        return tree.show(stdout=False)

    def _build_tree_structure(self, tree, parent_id=None, node_id_counter=[1]):
        node_id = node_id_counter[0]
        node_id_counter[0] += 1
        node_tag = (f"Example: {self.example.replace('\n', '\\n')}, "
                    f"Definition with prompt: {self.definition_with_prompt.replace('\n', '\\n')}, "
                    f"Definition without prompt: {self.definition_without_prompt.replace('\n', '\\n')}")
        tree.create_node(node_tag, node_id, parent=parent_id)
        for child in self.children:
            child._build_tree_structure(tree, node_id, node_id_counter)

@torch.no_grad()
def generate_examples_emb_gen(model, ex, tokenizerMLM, tokenizerTask, llama_model, llama_tokenizer, with_prompt=True, temperature=1):
    examples = ex['examples']
    print(examples)
    context = tokenizerMLM(examples, truncation=True, padding='longest', return_tensors='pt')
    nonce = "<nonce>"

    # Initialize the root of the tree with the initial examples and empty definitions
    root = TreeNode(example=examples[0], all_examples=examples.copy(), definition_with_prompt="", definition_without_prompt="")
    
    # Use a queue to perform BFS
    queue = [(root, 0)]  # (node, current_layer)
    
    while queue:
        current_node, current_layer = queue.pop(0)
        
        if current_layer < 3:  # Limit to 3 layers
            generated_exs = []
            try:
                context = tokenizerMLM(current_node.all_examples, truncation=True, padding='longest', return_tensors='pt')
            except Exception as e:
                print("An error occurred during tokenization. Here are the examples:")
                print(current_node.all_examples)
                raise e  # Re-throw the error to handle it further up the call stack or halt the program
            
            for _ in range(3):  # Generate 3 children per node
                # Generate example using LLaMA model and tokenizer
                prompt = example_prompt.format("\n".join(current_node.all_examples), nonce)
                # inputs = tokenizerTask(prompt, truncation=True, return_tensors='pt', max_length=256).to(device_0)
                # outputs = generate(model, context, inputs['input_ids'], inputs['attention_mask'], 30, mask_new_tokens=False, temperature=10)
                # gen_ex = tokenizerTask.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
                gen_ex = generate_text_completion(llama_model, llama_tokenizer, prompt, temperature=temperature)
                # Check if gen_ex contains the substring "<nonce>"
                if "<nonce>" not in gen_ex:
                    gen_ex = "sentence without <nonce>"
                # Append the generated example to the list
                current_node.all_examples.append(gen_ex)
                # gen_def_with_prompt = generate_definition(model, current_node.all_examples, tokenizerMLM, tokenizerTask, with_prompt=True)
                # gen_def_without_prompt = generate_definition(model, current_node.all_examples, tokenizerMLM, tokenizerTask, with_prompt=False)
                try:
                    gen_def_with_prompt = generate_definition(model, [gen_ex], tokenizerMLM, tokenizerTask, with_prompt=True)
                    gen_def_without_prompt = generate_definition(model, [gen_ex], tokenizerMLM, tokenizerTask, with_prompt=False)
                except Exception as e:
                    print(f"An error occurred: {e}")
                    print(f"gen_ex: {gen_ex}")
                    # throw the error to handle it further up the call stack or halt the program
                    raise e
                # Remove the generated example from the list
                current_node.all_examples.pop()
                generated_exs.append({
                    'example': gen_ex,
                    'definition_with_prompt': gen_def_with_prompt,
                    'definition_without_prompt': gen_def_without_prompt
                })
            
            # Create tree nodes for each generated example and add them as children to the current node
            for gen_ex in generated_exs:
                new_node = TreeNode(
                    example=gen_ex['example'],
                    all_examples=current_node.all_examples + [gen_ex['example']],
                    definition_with_prompt=gen_ex['definition_with_prompt'],
                    definition_without_prompt=gen_ex['definition_without_prompt']
                )
                current_node.add_child(new_node)
                queue.append((new_node, current_layer + 1))
    print(root)
    return root

def generate_examples_from_node(model, node, examples, tokenizerMLM, tokenizerTask, llama_model, llama_tokenizer, temperature):
    examples.append(node.example)
    prompt = example_prompt.format("\n".join(examples), "<nonce>")
    for _ in range(3):  # Loop for 3 generations
        gen_ex = generate_text_completion(llama_model, llama_tokenizer, prompt, temperature=temperature)
        examples.append(gen_ex)
        gen_def_with_prompt = generate_definition(model, examples, tokenizerMLM, tokenizerTask, with_prompt=True)
        gen_def_without_prompt = generate_definition(model, examples, tokenizerMLM, tokenizerTask, with_prompt=False)
        examples.pop()
        new_node = TreeNode(
            example=gen_ex,
            definition_with_prompt=gen_def_with_prompt,
            definition_without_prompt=gen_def_without_prompt
        )
        node.add_child(new_node)
        generate_examples_from_node(model, new_node, examples, tokenizerMLM, tokenizerTask, llama_model, llama_tokenizer, temperature)
    examples.pop()

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
    id = f"{timestamp}"
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
    batch_outputs = []  # Temporary list to store actual tree objects for every batch
    string_outputs = []  # List to store string representations of each tree

    print('Using llama-3')

    for i, ex in enumerate(def_task):
        if i >= 1:  # Break the loop after processing 1 example
            break
        print(f"Processing example {i + 1}...")
        print(ex)
        # Generate examples with prompt but the definition without prompt
        tree = generate_examples_emb_gen(model, ex, tokenizerMLM, tokenizerTask, model_llama3, tokenizer_llama3, with_prompt=True, temperature=temperature)
        # Save the tree
        batch_outputs.append(tree)
        # Save the string representation of the tree
        string_outputs.append(str(tree))
        treelib_tree = tree.__str__(return_tree_object=True)
        treelib_tree.to_graphviz(f"graphviz.dot")

    # After breaking out of the loop, save the processed examples
    print("Saving processed examples...")

    # Save the string representations to a CSV file
    df = pd.DataFrame({'tree_string': string_outputs})
    df.to_csv(fname_format + '.csv', mode='w', header=True, index=False)
    print('Saved string representations to:', fname_format + '.csv')

    # Save the actual tree objects to a pickle file
    with open(fname_format + '.pkl', 'wb') as f:
        pickle.dump(batch_outputs, f)
    print('Saved actual tree objects to:', fname_format + '.pkl')

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
