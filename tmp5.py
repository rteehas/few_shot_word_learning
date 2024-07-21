import pandas as pd
import evaluate

# Load the BLEU, ROUGE, and BERTScore metrics
bleu_metric = evaluate.load('bleu')
rouge_metric = evaluate.load('rouge')
bertscore_metric = evaluate.load('bertscore')

# Function to calculate BLEU score (unchanged)
def calculate_bleu(reference, candidate):
    results = bleu_metric.compute(predictions=[candidate], references=[[reference]])
    return results['bleu']

# Function to calculate ROUGE score (unchanged)
def calculate_rouge(reference, candidate):
    results = rouge_metric.compute(predictions=[candidate], references=[reference])
    return results['rougeL']

# Function to calculate BERTScore
def calculate_bertscore(reference, candidate):
    # Specify the device as 'cuda' to use the GPU
    results = bertscore_metric.compute(predictions=[candidate], references=[reference], lang='en', device='cuda', model_type='microsoft/deberta-large-mnli')
    return results['f1'][0]  # results['f1'] is a list, get the first element for the current pair

# Read the CSV file (unchanged)
df = pd.read_csv('subset.csv')

# Calculate BLEU, ROUGE, and BERTScore for each row and add them as new columns
# df['BLEU'] = df.apply(lambda row: calculate_bleu(row['definition'], row['generated definition']), axis=1)
# df['ROUGE'] = df.apply(lambda row: calculate_rouge(row['definition'], row['generated definition']), axis=1)
df['BERTScore_F1'] = df.apply(lambda row: calculate_bertscore(row['definition'], row['generated definition']), axis=1)

# Save the DataFrame with the new columns to a new CSV file
df.to_csv('subset_scores.csv', index=False)