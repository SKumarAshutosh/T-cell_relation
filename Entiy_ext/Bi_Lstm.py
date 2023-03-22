import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

# Define the device to use for the model (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the lists of entities to extract
t_cells = ["T-cells", "T cells"]
cytokines = ["cytokine", "cytokines"]
transcription_factors = ["transcription factor", "transcription factors"]

# Define the BiLSTM model
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output

# Load the pre-trained model
model = BiLSTM(input_dim=300, hidden_dim=256, output_dim=3)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)

# Define a function to extract entities from a single sentence
def extract_entities(sentence):
    # Convert the sentence to a tensor
    sentence = torch.tensor(sentence).unsqueeze(0).to(device)

    # Get the output from the model
    output = model(sentence)

    # Get the predicted entity label
    predicted_label = torch.argmax(output, dim=2).squeeze().item()

    # Determine the entity type based on the predicted label
    entity_type = None
    if predicted_label == 0:
        entity_type = "T-cell"
    elif predicted_label == 1:
        entity_type = "Cytokine"
    elif predicted_label == 2:
        entity_type = "Transcription factor"

    # Extract the entity from the sentence
    entity = None
    if entity_type == "T-cell":
        for t_cell in t_cells:
            if t_cell.lower() in sentence.lower():
                entity = t_cell
                break
    elif entity_type == "Cytokine":
        for cytokine in cytokines:
            if cytokine.lower() in sentence.lower():
                entity = cytokine
                break
    elif entity_type == "Transcription factor":
        for transcription_factor in transcription_factors:
            if transcription_factor.lower() in sentence.lower():
                entity = transcription_factor
                break

    return (entity, entity_type, sentence)

# Define the list of biomedical texts to extract entities from
texts = [
    "T-cells are important for immune function.",
    "IL-10 is an anti-inflammatory cytokine.",
    "NF-kappaB is a transcription factor.",
    "IL-2 is produced by T cells."
]

# Extract entities from each text and store the results in a list of tuples
results = []
for text in texts:
    results.append(extract_entities(text))

# Store the results in a CSV file
df = pd.DataFrame(results, columns=["Entity", "Type", "Sentence"])
df.to_csv("output.csv", index=False)

# Print the results
print(results)