from datasets import load_dataset

# Specify the path to your dataset.arrow file
dataset_path = "/scratch/rst306/few_shot_repo/definition_task_outputs/emb_gen_generations_masked_new_token_new_data_new_model/dataset.arrow"

# Load the dataset
dataset = load_dataset('arrow', data_files=dataset_path)

# Now you can work with the dataset
print(dataset)

# Filter the dataset to keep only examples where 'prompt' starts with "The word" and len(example['examples']) == 1
subset = dataset['train'].filter(lambda example: example['prompt'].startswith("The word") and len(example['examples']) == 1)

# Now, 'subset' contains the filtered dataset
print(subset)

# Save the subset to an Arrow file
subset.save_to_disk("subset.arrow")

# Print a confirmation message
print("Subset saved to Arrow file successfully.")