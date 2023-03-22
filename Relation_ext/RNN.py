import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Set the named entities
entities = [
    "T-cell",
    "cytokines",
    "transcription factor",
]

# Set the sentences to be used for training
sentences = [
    "T-cells produce cytokines",
    "Cytokines activate transcription factors",
    "Transcription factors regulate T-cell function",
]

# Set the relations and their types
relations = [
    ("T-cell", "cytokines", "produce"),
    ("cytokines", "transcription factor", "activate"),
    ("transcription factor", "T-cell", "regulate"),
]
relation_types = [
    "production",
    "activation",
    "regulation",
]

# Tokenize the words in the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(entities + sentences)
entity_sequences = tokenizer.texts_to_sequences(entities)
sentence_sequences = tokenizer.texts_to_sequences(sentences)

# Pad the sequences to be of equal length
entity_sequences = pad_sequences(entity_sequences, maxlen=len(entities))
sentence_sequences = pad_sequences(sentence_sequences, maxlen=len(entities))

# Create the training data
X = np.concatenate((entity_sequences, sentence_sequences), axis=1)
y = np.zeros((len(relations), len(entities), len(entities), len(relation_types)))
for i, (e1, e2, relation) in enumerate(relations):
    e1_idx = entities.index(e1)
    e2_idx = entities.index(e2)
    relation_idx = relation_types.index(relation)
    y[i, e1_idx, e2_idx, relation_idx] = 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the RNN model
model = Sequential()
model.add(Embedding(input_dim=len(entities) + len(sentences) + 1, output_dim=32, input_length=len(entities) + len(sentences)))
model.add(SimpleRNN(units=16))
model.add(Dense(units=len(entities) * len(entities) * len(relation_types), activation="softmax"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=100, validation_data=(X_test, y_test))

# Predict the relations on new data
new_entities = [
    "cytokines",
    "T-cell",
    "transcription factor",
]
new_sentence = "T-cells and cytokines interact to activate transcription factors"
new_entity_sequences = pad_sequences(tokenizer.texts_to_sequences(new_entities), maxlen=len(entities))
new_sentence_sequences = pad_sequences(tokenizer.texts_to_sequences([new_sentence]), maxlen=len(sentences))
new_X = np.concatenate((new_entity_sequences, new_sentence_sequences), axis=1)
predicted_relations = model.predict(new_X)

# Print the predicted relations and their types
for e1_idx, e1 in enumerate(entities):
    for e2_idx, e2 in enumerate(entities):
        for relation_idx, relation_type in enumerate(relation_types):
            if predicted_relations[0, e1_idx, e2
            if predicted_relations[0, e1_idx, e2_idx, relation_idx] > 0.5:
                print(f"{e1} {relation_type} {e2}")