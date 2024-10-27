import torch
from flask import Flask,  jsonify
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
import pandas as pd
import requests
from flask import Flask
import torchtext

class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_units, num_classes):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_units)
        self.rnn = nn.GRU(hidden_units, hidden_units, batch_first=True)
        self.fc = nn.Linear(hidden_units, num_classes)

    def forward(self,x):
        embedded = self.embedding(x)
        out, _= self.rnn(embedded)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


model = RNN()
model.load_state_dict(torch.load("", weights_only=True), strict=False)
model.eval()


app = Flask(__name__)
@app.route('/send-string', methods=['POST'])
def index():
    return jsonify({ "prediction": max_probability })
if __name__ == '__main__':
    app.run(port=5000, debug=True)

emailContents = ""
def fetch_data():
    input = requests.get('http://localhost:5000/flask')
    if input.status_code == 200:
        data = input.json()
        return data
if  __name__ == '__main__':
    emailContents = fetch_data()
    
app = Flask(__name__)
@app.route('/flask',methods=['POST'])
def index():
    return "Flask server"

if __name__ == "__main__":
    app.run(port=5000,debug=True)

text = torchtext.data.Field(sequential=True,
                            tokenize=lambda x: x,
                            include_lengths=True,
                            batch_first=True,
                            use_vocab=True)

def predict_string(input_string):
    # Tokenize the input string
    tokenized_input = [char for char in input_string]  # Adjust this based on your tokenizer
    # Convert to tensor
    input_tensor = text.process([tokenized_input])
    
    # Step 4: Make a prediction
    with torch.no_grad():  # Disable gradient calculation
        prediction = model(input_tensor)
        prob = nn.functional.softmax(prediction, dim=1)
        max_index = torch.argmax(prob, dim=1)
        max_probability = prediction[0,max_index].item()

    return prediction       

    

