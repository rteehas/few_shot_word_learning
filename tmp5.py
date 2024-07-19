import pandas as pd
import evaluate

# Load the BLEU and ROUGE metrics
bleu_metric = evaluate.load('bleu')
rouge_metric = evaluate.load('rouge')

# Function to calculate BLEU score
def calculate_bleu(reference, candidate):
    # The BLEU metric in Hugging Face expects a list of predictions and a list of lists of references
    results = bleu_metric.compute(predictions=[candidate], references=[[reference]])
    return results['bleu']

# Function to calculate ROUGE score
def calculate_rouge(reference, candidate):
    # The ROUGE metric in Hugging Face expects a list of predictions and a list of references
    results = rouge_metric.compute(predictions=[candidate], references=[reference])
    # Returning the F1 score of ROUGE-L
    return results['rougeL']

# Read the CSV file
df = pd.read_csv('emb_gen_generations_masked_new_token_new_data_new_model.csv')

# Calculate BLEU and ROUGE scores for each row and add them as new columns
df['BLEU'] = df.apply(lambda row: calculate_bleu(row['definition'], row['generated definition']), axis=1)
df['ROUGE'] = df.apply(lambda row: calculate_rouge(row['definition'], row['generated definition']), axis=1)

# Save the DataFrame with the new columns to a new CSV file
df.to_csv('emb_gen_generations_masked_new_token_new_data_new_model_updated_with_scores.csv', index=False)