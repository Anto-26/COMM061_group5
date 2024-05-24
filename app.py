from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import pickle
from gensim.models import KeyedVectors
import datetime

app = Flask(__name__)

# Define the RNNModel class
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_dim, output_dim, n_layers=1, dropout=0.3):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_vectors):
        rnn_out, _ = self.rnn(input_vectors)
        output = self.fc(rnn_out)
        return output

# Define the load_model function
def load_model(file_path):
    with open(file_path, 'rb') as f:
        model_data = pickle.load(f)

    # Create the RNNModel instance
    model = RNNModel(model_data['input_size'], model_data['hidden_dim'], model_data['output_dim'])

    # Load the state_dict
    model.load_state_dict(model_data['state_dict'])

    # Retrieve word2vec_model and label_encoding
    word2vec_model = model_data['word2vec_model']
    label_encoding = model_data['label_encoding']

    return model, word2vec_model, label_encoding

# Load the model
model, word2vec_model, label_encoding = load_model('model.pkl')

# Function to prepare single input
def prepare_single_input(tokens, word2vec_model, max_len=128):
    word_vectors = [word2vec_model[token] if token in word2vec_model else np.zeros(word2vec_model.vector_size) for token in tokens]
    if len(word_vectors) > max_len:
        word_vectors = word_vectors[:max_len]
    else:
        pad_length = max_len - len(word_vectors)
        word_vectors.extend([np.zeros(word2vec_model.vector_size)] * pad_length)
    input_vector = np.array(word_vectors)
    input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
    return input_tensor

# Function to log user inputs, model predictions, and timestamps
def log_interaction(user_input, model_prediction):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} | User Input: {user_input} | Model Prediction: {model_prediction}\n"
    with open("interaction_log.txt", "a") as log_file:
        log_file.write(log_entry)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    tokens = data['tokens']

    # Prepare input tensor
    input_tensor = prepare_single_input(tokens, word2vec_model)

    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        preds = torch.argmax(outputs, dim=-1).squeeze().cpu().numpy()

    # Map numerical predictions to label strings
    predicted_labels = [list(label_encoding.keys())[pred] for pred in preds[:len(tokens)]]

    # Log the interaction
    log_interaction(tokens, predicted_labels)

    return jsonify({'predicted_labels': predicted_labels})

if __name__ == '__main__':
    app.run(debug=True)
