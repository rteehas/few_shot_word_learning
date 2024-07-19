from train_with_llama import *
from eval_wic import *
import einops
import json
import re


def new_embedding_prompt_completion(prompt, model, college_embedding, secondLM, tokenizerTask, completion_length=30):

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
    output_sequences = secondLM.generate(input_ids=inputs['input_ids'], max_length=len(inputs['input_ids'][0]) + completion_length)

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
    combined_model.eval()
    
    # prompt = "Q: What is the capital of Japan? \n A: Tokyo \nQ: What is the capital of Germany? \nA: Berlin \nQ: What is the capital of <nonce>? \nA:"
    prompt = "Q: <nonce> is the capital of which country? \nA:"

    # Initialize an empty dictionary for the results
    generated_texts = {}

    # Initialize a counter
    counter = 0

    # Loop through each country in the world_capitals dictionary
    for key, value in world_capitals.items():
        modified_examples = []
        replacement_made = False
        for example in value['capital_examples']:
            # Use regex for case-insensitive replacement
            modified_example, num_replacements = re.subn(re.escape(value['capital']), '<nonce>', example, flags=re.IGNORECASE)
            if num_replacements > 0:  # Check if replacement occurred
                replacement_made = True
            modified_examples.append(modified_example)
        if not replacement_made:  # If no replacement was made in any of the examples
            print(f"No replacement made for key: {key}, value: {value}")
            break
            # Exit the loop is handled outside this snippet
        examples = modified_examples  # Use the modified examples for further processing
        
        # Tokenize the modified examples
        tokenized_contexts = tokenizerMLM(examples, return_tensors='pt', padding='longest').to(device)
        
        generated_text = None

        try:
            # Get college embeddings
            _, _, college_embeds = model.get_college_embeddings([tokenized_contexts])
            # Generate text using the new embedding
            generated_text = new_embedding_prompt_completion(prompt, model, college_embeds[0], secondLM, tokenizerTask, completion_length=10)
        except Exception as e:
            print(f"An error occurred for key: {key}. Error: {e}")
            break
            

        # Check if generated_text contains the capital
        if value['country'] in generated_text:
            print(f"Key: {key}, Value: {value}, Generated Text: {generated_text}")

        # Save the generated text in the new dictionary using the same key
        generated_texts[key] = generated_text

        # Increment the counter
        counter += 1

        # # # Break the loop after 2 iterations
        # if counter == 2:
        #     break

    # At this point, generated_texts contains the generated text for each country, keyed by the same keys as in world_capitals
    # save generated_texts to a json
    with open('scratch_outputs/world_capitals_capitals_output.json', 'w') as file:
        json.dump(generated_texts, file, indent=4)
