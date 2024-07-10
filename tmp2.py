from train_with_llama import *
from eval_wic import *
import einops
import json


def new_embedding_prompt_completion(prompt, model, college_embedding, secondLM, tokenizerTask):

    input_embeds, output_embeds = model.emb_gen.get_input_and_output_embedding(college_embedding) 

    input_weight = combined_model.get_new_weights(task='Task', new_embed=input_embeds)
    output_weight = combined_model.get_new_output_weights(new_embed=output_embeds)


    secondLM.eval()

    embedding_layer = secondLM.model.embed_tokens

    old_num_tokens, old_embedding_dim = embedding_layer.weight.shape

    num_new_tokens = 1

    old_embedding_weights = secondLM.model.embed_tokens.weight.clone()
    old_lm_head_weights = secondLM.lm_head.weight.clone()

    new_embeddings = nn.Embedding(old_num_tokens + num_new_tokens, old_embedding_dim)

    new_embeddings.to(embedding_layer.weight.device, dtype=embedding_layer.weight.dtype)

    new_embeddings.weight.data = input_weight

    secondLM.model.embed_tokens = new_embeddings
    secondLM.lm_head = torch.nn.Linear(old_embedding_dim, old_num_tokens + num_new_tokens, bias=False)
    secondLM.lm_head.weight.data = output_weight
    inputs = tokenizerTask(prompt, truncation=True, return_tensors='pt', max_length=256).to(device)

    # Generate text
    output_sequences = secondLM.generate(input_ids=inputs['input_ids'], max_length=len(inputs['input_ids'][0]) + 30)

    # Decode generated text
    generated_text = tokenizerTask.decode(output_sequences[0], skip_special_tokens=False)

    secondLM.model.embed_tokens = nn.Embedding(old_num_tokens, old_embedding_dim)
    secondLM.model.embed_tokens.weight.data = old_embedding_weights
    secondLM.lm_head = torch.nn.Linear(old_embedding_dim, old_num_tokens, bias=False)
    secondLM.lm_head.weight.data = old_lm_head_weights
    return generated_text

def process_embeddings(prompt, model, secondLM, tokenizerTask, college_embedding_A, college_embedding_B, college_embedding_C, college_embedding_D):
    # Calculate C_approx and D_approx
    D_approx = college_embedding_B - college_embedding_A + college_embedding_C
    C_approx = college_embedding_A - college_embedding_B + college_embedding_D

    # List of embeddings to process
    embeddings = [
        ("A", college_embedding_A),
        ("B", college_embedding_B),
        ("C", college_embedding_C),
        ("D", college_embedding_D),
        ("C_approx = A - B + D", C_approx),
        ("D_approx = B - A + D", D_approx)
    ]

    # Dictionary to store results
    results = {}

    # Process each embedding
    for name, embedding in embeddings:
        generated_text = new_embedding_prompt_completion(prompt, model, embedding, secondLM, tokenizerTask)
        results[name] = generated_text

    return results

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    path = "/scratch/jl16973/college_pretrained_model/checkpoint_7_28000"

    firstLM = RobertaForMaskedLM.from_pretrained("roberta-large", low_cpu_mem_usage=True).to(device)
    secondLM = LlamaForCausalLM.from_pretrained("/vast/work/public/ml-datasets/llama-2/Llama-2-7b-hf", low_cpu_mem_usage=True).to(device)
    tokenizerTask = LlamaTokenizer.from_pretrained(path + "tokenizerTask", use_fast=False, legacy=True)
    tokenizerMLM = AutoTokenizer.from_pretrained(path + "/tokenizerMLM", use_fast=False)
    memory_config = AggregatorConfig()
    nonces = list(tokenizerTask.get_added_vocab().keys())
    print(nonces)
    mask_token_id = tokenizerMLM.mask_token_id
    layers=[-1]

    new_token_idx = len(tokenizerTask) - 1
    # tokenizerMLM.add_tokens(["<nonce>"])

    with open('world_capitals.json') as file:
        world_capitals = json.load(file)
    
    examples_A = world_capitals['Canada-Ottawa']['country_examples']
    examples_A = [example.replace(world_capitals['Canada-Ottawa']['country'], '<nonce>') for example in examples_A]

    examples_B = world_capitals['France-Paris']['country_examples']
    examples_B = [example.replace(world_capitals['France-Paris']['country'], '<nonce>') for example in examples_B]

    examples_C = world_capitals['Canada-Ottawa']['country_examples'][:2]
    examples_C = [example.replace(world_capitals['Canada-Ottawa']['country'], '<nonce>') for example in examples_C]

    examples_D = world_capitals['United Kingdom-London']['country_examples']
    examples_D = [example.replace(world_capitals['United Kingdom-London']['country'], '<nonce>') for example in examples_D]





    model = CoLLEGeEmbeddingModel(
        firstLM=firstLM,
        num_new_tokens=1,
        layers=layers,
        mask_token_id=mask_token_id,
        memory_config=memory_config,
        num_layers=1,
        distillation_temp=0.3,
        use_pos=False,
    ).to(device)
    model.emb_gen.load_state_dict(torch.load(path + "/pytorch_model.bin"))
    model.eval()

    combined_model = MorphMemoryModelLLAMA(
        firstLM=firstLM,
        secondLM=secondLM,
        num_new_tokens=1,
        layers=layers,
        mask_token_id=mask_token_id,
        memory_config=memory_config,
        num_layers=1,
        distillation_temp=0.3,
    )
    combined_model.emb_gen.load_state_dict(torch.load(path + "/pytorch_model.bin"))

    tokenized_contexts_A = tokenizerMLM(examples_A, return_tensors='pt', padding='longest').to(device)
    _, _, college_embeds_A = model.get_college_embeddings([tokenized_contexts_A])
    
    tokenized_contexts_B = tokenizerMLM(examples_B, return_tensors='pt', padding='longest').to(device)
    _, _, college_embeds_B = model.get_college_embeddings([tokenized_contexts_B])

    tokenized_contexts_C = tokenizerMLM(examples_C, return_tensors='pt', padding='longest').to(device)
    _, _, college_embeds_C = model.get_college_embeddings([tokenized_contexts_C])

    tokenized_contexts_D = tokenizerMLM(examples_D, return_tensors='pt', padding='longest').to(device)
    _, _, college_embeds_D = model.get_college_embeddings([tokenized_contexts_D])

    # Assuming B and C are defined or given expressions/values
    college_embeds = college_embeds_A[0] 
    
    input_embeds, output_embeds = model.emb_gen.get_input_and_output_embedding(college_embeds) 
    # einops.rearrange(input_embeds, 'x -> 1 x')
    # einops.rearrange(output_embeds, 'x -> 1 x')
    print('colleged_embeds: ', college_embeds.shape)
    print('input_embeds: ', input_embeds.shape)
    print('output_embeds: ', output_embeds.shape)


    # input_weight = combined_model.get_new_weights(task = 'Task', new_embed=input_embeds)
    # output_weight = combined_model.get_new_output_weights(new_embed=output_embeds)

    # # print the shapes
    # print('input_embeds: ', input_weight.shape)
    # print('output_embeds: ', output_weight.shape)

    # secondLM.eval()
    # # Get the model's state dict

    # print(secondLM.model.embed_tokens.weight.size()) 
    # embedding_layer = secondLM.model.embed_tokens

    # old_num_tokens, old_embedding_dim = embedding_layer.weight.shape

    # num_new_tokens = 1

    # # Creating new embedding layer with more entries
    # new_embeddings = nn.Embedding(
    #         old_num_tokens + num_new_tokens, old_embedding_dim
    # )

    # # Setting device and type accordingly
    # new_embeddings.to(
    #     embedding_layer.weight.device,
    #     dtype=embedding_layer.weight.dtype,
    # )

    # new_embeddings.weight.data = input_weight

    # secondLM.model.embed_tokens = new_embeddings
    # print(secondLM.model.embed_tokens)
    # secondLM.lm_head = torch.nn.Linear(old_embedding_dim, old_num_tokens + num_new_tokens, bias=False)
    # secondLM.lm_head.weight.data = output_weight
    # print('lm_head: ', secondLM.lm_head)


    # Print all layers of the model
    # for name, module in secondLM.named_modules():
    #     print(name, module)

    prompt = "Q: What is the capital of <nonce>? \nA:"

    # prompt = """
    # The word "<nonce>" is defined as
    # """
    # inputs = tokenizerTask(prompt, truncation=True, return_tensors='pt', max_length=256).to(device)

    # # Generate text
    # output_sequences = secondLM.generate(input_ids=inputs['input_ids'], max_length=512)

    # # Decode generated text
    # print('output_sequences: ', output_sequences)
    # generated_text = tokenizerTask.decode(output_sequences[0], skip_special_tokens=True)
    # print(generated_text)

    # generated_text = new_embedding_prompt_completion(prompt, model, college_embeds, secondLM, tokenizerTask)

    generated_text = process_embeddings(prompt, model, secondLM, tokenizerTask, college_embeds_A[0], college_embeds_B[0], college_embeds_C[0], college_embeds_D[0])
    print(generated_text)

    # Assuming generated_text is the dictionary returned by the process_embeddings function
    # for key in sorted(generated_text.keys()):
    #     print('-----------------')
    #     print(f"{key}: {generated_text[key]}")

    # # decode token id 32000
    # print(tokenizerTask.decode(torch.tensor([32000], device=device)))

    # print('-----------------')
    # print('outputs from generate')
    # outputs = generate(combined_model, tokenized_contexts_A, inputs['input_ids'], inputs['attention_mask'], 30, mask_new_tokens=True)
    # # print(outputs)
    # generated_def = tokenizerTask.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
    # print(generated_def)

