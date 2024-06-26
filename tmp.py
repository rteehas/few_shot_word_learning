from datasets import load_from_disk
import pandas as pd


# Assuming fname_format is the directory where the dataset was saved
dataset = load_from_disk('/scratch/rst306/few_shot_repo/definition_task_outputs/emb_gen_generations')

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(dataset)

# Export the DataFrame to a CSV file
csv_file_path = 'emb_gen_generations.csv'
df.to_csv(csv_file_path, index=False)

print(f"Dataset exported to {csv_file_path}")