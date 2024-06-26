import json

# Initialize an empty dictionary for the dataset
dataset = {}

# Open and read the file
with open('E10-male-female.txt', 'r') as file:
    for line in file:
        # Split each line into words, considering possible multiple counterparts but only taking the first
        words = line.strip().split('\t')
        key_word = words[0]
        first_counterpart = words[1].split('/')[0]  # Only take the first counterpart
        
        # Create a unique key for the pair to maintain the hierarchical relationship
        pair_key = f"{key_word}-{first_counterpart}"
        
        # Initialize the pair in the dataset with an ordered list and sub-keys for each word
        dataset[pair_key] = {
            'order': [key_word, first_counterpart],
            key_word: {'sentences': []},
            first_counterpart: {'sentences': []}  # Only include the first counterpart
        }

# Convert the dictionary to JSON and write it to a file
with open('dataset2.json', 'w') as json_file:
    json.dump(dataset, json_file, indent=4)