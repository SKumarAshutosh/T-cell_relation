import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load the BioBERT model and tokenizer
model_name = 'dmis-lab/biobert-large-cased-v1.1'
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the entity types we want to extract
entity_types = ['T-cell', 'cytokine', 'transcription factor']

# Define a dictionary to store the results
results = {entity_type: [] for entity_type in entity_types}

# Loop through each sentence in the text
for sentence in sentences:
    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence)
    # Add the [CLS] and [SEP] tokens
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    # Convert the tokens to their corresponding IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Convert the token IDs to a PyTorch tensor
    token_tensor = torch.tensor([token_ids])
    # Run the token tensor through the model
    outputs = model(token_tensor)
    # Get the predicted label for each token
    _, predicted_labels = torch.max(outputs[0], 2)
    predicted_labels = predicted_labels.squeeze().tolist()
    # Extract the entities from the sentence
    for entity_type in entity_types:
        entity_tokens = []
        entity = False
        for i, label in enumerate(predicted_labels):
            if i == 0 or i == len(predicted_labels) - 1:
                continue
            if label == entity_label_map[entity_type]['B']:
                entity_tokens.append(tokens[i])
                entity = True
            elif label == entity_label_map[entity_type]['I'] and entity:
                entity_tokens.append(tokens[i])
            else:
                if entity:
                    entity_string = tokenizer.convert_tokens_to_string(entity_tokens)
                    results[entity_type].append((entity_string, entity_type, sentence))
                    entity_tokens = []
                    entity = False

# Write the results to a CSV file
with open('bioentities.csv', 'w') as f:
    f.write('Entity,Type,Sentence\n')
    for entity_type in entity_types:
        for entity in results[entity_type]:
            f.write(f'{entity[0]},{entity[1]},{entity[2]}\n')