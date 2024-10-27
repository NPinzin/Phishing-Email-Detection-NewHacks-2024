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

emailContents = " "
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
label = torchtext.data.Field(sequential=False,
                            use_vocab=False,
                            is_target=True,
                            batch_first=True,
                            dtype = torch.float)

fields = [('text', text), ('label', label)]
dataset = torchtext.data.TabularDataset("RNN/SpamHam/spam_ham_datasets.csv","csv",fields, skip_header=True)
train, validate, test = dataset.split(split_ratio=[0.6,0.2,0.2])
train_iter = torchtext.data.BucketIterator(train,
                                           batch_size=32,
                                           sort_key=lambda x: len(x.text), # to minimize padding
                                           sort_within_batch=True,        # sort within each batch
                                           repeat=False)                  # repeat the iterator for many epochs
text.build_vocab(train)
model = RNN(len(text.vocab.itos), 128, 1)
model.load_state_dict(torch.load("RNN/saved_parameters/best_initial_model.pth", weights_only=True), strict = False)

with torch.no_grad():  # Disable gradient calculation
    prediction = model(emailContents)
    prob = nn.functional.softmax(prediction, dim=1)
    max_index = torch.argmax(prob, dim=1)
    max_probability = prediction[0,max_index].item()


    

