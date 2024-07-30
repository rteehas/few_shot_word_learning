import torch
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast

def load_model_and_tokenizer(model_name, device):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer

def generate_text_completion(model, tokenizer, prompt, max_length=50, temperature=1.0):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    output = model.generate(input_ids, max_length=max_length, temperature=temperature, do_sample=True)
    completion = tokenizer.decode(output[0], skip_special_tokens=True)
    return completion

# Check the number of available GPUs
num_gpus = torch.cuda.device_count()
print(f"Number of available GPUs: {num_gpus}")

if num_gpus >= 2:
    # Load LLaMA-3 on GPU 0 from the local path
    device_0 = torch.device('cuda:0')
    model_path_llama3 = '/vast/work/public/ml-datasets/llama-3/Meta-Llama-3-8B-hf'
    model_llama3, tokenizer_llama3 = load_model_and_tokenizer(model_path_llama3, device_0)

    # Load LLaMA-2 on GPU 1
    device_1 = torch.device('cuda:1')
    model_name_llama2 = '/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf'  
    model_llama2, tokenizer_llama2 = load_model_and_tokenizer(model_name_llama2, device_1)

    # Example usage
    prompt = "Once upon a time"

    # Generate text using LLaMA-3 on GPU 0
    completion_llama3 = generate_text_completion(model_llama3, tokenizer_llama3, prompt)
    print(f"LLaMA-3 completion: {completion_llama3}")

    # Generate text using LLaMA-2 on GPU 1
    completion_llama2 = generate_text_completion(model_llama2, tokenizer_llama2, prompt)
    print(f"LLaMA-2 completion: {completion_llama2}")
else:
    print("Not enough GPUs available.")