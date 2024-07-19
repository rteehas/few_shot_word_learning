from datasets import load_dataset
import csv
import json
import itertools
import re
from concurrent.futures import ProcessPoolExecutor

# Load the Wikipedia dataset, specify the version or configuration if needed
dataset = load_dataset("wikipedia", "20220301.en")

def extract_sentences(text, country, max_sentences=10):
    sentences = re.split(r'\. |\.\n', text)
    matching_sentences = []
    for sentence in sentences:
        if country.lower() in sentence.lower():
            matching_sentences.append(sentence)
            if len(matching_sentences) == max_sentences:
                break
    return matching_sentences

def process_row(row):
    country = row['country']
    capital = row['capital']
    country_capital_pair = {
        'country': country,
        'capital': capital,
        'country_examples': [],
        'capital_examples': []
    }
    articles = dataset['train'].filter(lambda example: country.lower() in example['title'].lower())
    for article in articles:
        if country.lower() in article['title'].lower():
            sentences = extract_sentences(article['text'], country)
            country_capital_pair['country_examples'].extend(sentences)
            if len(country_capital_pair['country_examples']) >= 10:
                country_capital_pair['country_examples'] = country_capital_pair['country_examples'][:10]
                break

    capital_articles = dataset['train'].filter(lambda example: capital.lower() in example['title'].lower())
    for article in capital_articles:
        if capital.lower() in article['title'].lower():
            sentences = extract_sentences(article['text'], capital)
            country_capital_pair['capital_examples'].extend(sentences)
            if len(country_capital_pair['capital_examples']) >= 10:
                country_capital_pair['capital_examples'] = country_capital_pair['capital_examples'][:10]
                break
            
    return country_capital_pair

def main():
    csv_file_path = '../lm_vector_arithmetic/world_capitals.csv'
    country_capital_pairs = {}

    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # Convert to list to slice for demonstration

    with ProcessPoolExecutor() as executor:
        results = executor.map(process_row, rows)

    for result in results:
        key = f"{result['country']}-{result['capital']}"
        country_capital_pairs[key] = result

    json_file_path = 'world_capitals.json'
    with open(json_file_path, 'w') as file:
        json.dump(country_capital_pairs, file, indent=4)

if __name__ == "__main__":
    main()