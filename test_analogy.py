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
    generated_text = tokenizerTask.decode(output_sequences[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

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
    
    examples_A = [
        'The <nonce> decided to shave his beard for the first time in a year, revealing a strong jawline beneath.',
        'Every morning, the <nonce> would lift weights in the gym, pushing his limits to gain muscle.',
        'The <nonce> taught his son how to fix the car, passing down knowledge from one generation to the next.',
        'In the face of adversity, the <nonce> stood firm, showing great courage and resilience.'
    ]

    examples_B = [
        'The <nonce> decided to let her hair grow long, embracing its natural waves and texture.',
        'Every morning, the <nonce> would practice yoga, finding strength and flexibility in each pose.',
        'The <nonce> taught her daughter how to bake, passing down family recipes filled with love and tradition.',
        'In the face of adversity, the <nonce> stood with grace, showing great empathy and understanding.'
    ]

    examples_C = [
        'The <nonce> sharpened his swordsmanship, ready to defend his kingdom as the valiant son of the king and queen.',
        'In the vast library of the castle, the <nonce> poured over ancient texts, seeking the wisdom needed to rule wisely after his parents.',
        'The <nonce> rode through the kingdom on horseback, showing the people the strength and courage he had inherited from his royal lineage.',
        'At state functions, the <nonce> displayed impeccable manners and a keen understanding of politics, traits befitting the heir to the throne.'
    ]

    examples_D = [
        'The <nonce> wore her crown with pride, knowing she was the beloved daughter of the king and queen.',
        'In the royal garden, the <nonce> learned the art of diplomacy, preparing to one day lead her people with wisdom.',
        'The <nonce> studied the history of her kingdom, eager to honor the legacy of her parents and serve her subjects faithfully.',
        'At the grand ball, the <nonce> danced elegantly, her regal presence a testament to her royal upbringing.'
    ]

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

    with open('dataset_male-female.json', 'r') as dataset_file:
        data = json.load(dataset_file)

    prompt = """
Q: The gender of a king is?
A: Male

Q: The gender of a queen is?
A: Female

Q: The gender of a waitress is?
A: Female

Q: The gender of a waiter is?
A: Male

Q: The gender of a <nonce> is?
"""

    # prompt = """
    # The word "<nonce>" is defined as
    # """
    generated_text = process_embeddings(prompt, model, secondLM, tokenizerTask, college_embeds_A[0], college_embeds_B[0], college_embeds_C[0], college_embeds_D[0])

    results = {}  # Initialize an empty dictionary to store results

    for index, (word_pair, details) in enumerate(data.items()):
        # Uncomment the next two lines to limit the loop to 10 iterations
        # if index >= 2:  
        #     break  

        print(f"Processing word pair: {word_pair}")
        order = details['order']
        
        examples_C = details[order[0]]['sentences']
        examples_D = details[order[1]]['sentences']
            
        tokenized_contexts_C = tokenizerMLM(examples_C, return_tensors='pt', padding='longest').to(device)
        _, _, college_embeds_C = model.get_college_embeddings([tokenized_contexts_C])

        tokenized_contexts_D = tokenizerMLM(examples_D, return_tensors='pt', padding='longest').to(device)
        _, _, college_embeds_D = model.get_college_embeddings([tokenized_contexts_D])
        generated_text = process_embeddings(prompt, model, secondLM, tokenizerTask, college_embeds_A[0], college_embeds_B[0], college_embeds_C[0], college_embeds_D[0])
        
        # Save the generated text in the results dictionary
        result = {}
        result['generated_text'] = generated_text
        result['words'] = {
            "A": "man",
            "B": "woman",
            "C": order[0],
            "D": order[1]
        }
        result['example_sentences'] = {
            "A": examples_A,
            "B": examples_B,
            "C": examples_C,
            "D": examples_D
        }
        results[word_pair] = result
        
        print(f"Generated text: \n{generated_text}\n")
        print("---------------------------\n")

    # After the loop, save the results dictionary to a JSON file
    with open('results_dataset.json', 'w') as file:
        json.dump(results, file, indent=4)
