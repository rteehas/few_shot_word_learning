#!/bin/env python

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Clear GPU memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load model directly
llama_path = "/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(llama_path, low_cpu_mem_usage=True)
model = AutoModelForCausalLM.from_pretrained(llama_path, low_cpu_mem_usage=True)

# Move model to GPU if available
if torch.cuda.is_available():
    model = model.to('cuda')

# Tokenize input text
input_text = "The definition of classical music is"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Move input to GPU if available
if torch.cuda.is_available():
    input_ids = input_ids.to('cuda')

# Generate text
output = model.generate(input_ids, max_length=100, temperature=0.7, do_sample=True)

# Decode the output
output_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(output_text)