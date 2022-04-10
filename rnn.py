from sys import getsizeof

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from tqdm import tqdm

from dataset import Tab22_Dataset

import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

submit = True
no = 1
train_split = 0.8

# Hyperparameters
input_size = 13
sequence_length = 60
num_layers = 2
hidden_size = 256
num_classes = 2
learning_rate = 0.01
batch_size = 64
num_epochs = 120

# Model 
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        out = nn.Softmax(out)
        return out

class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        out = F.log_softmax(out)
        return out

    


def write_submission(model, no):

    X_csv = pd.read_csv('./data/test.csv')

    X = torch.tensor(X_csv.values[:,3:], dtype=torch.float32).reshape(-1, 60, 13).to(device=device)
    sequence = torch.arange(25968, 38186).to('cpu')


    predictions = []

    print('Writing ... ')
    for i in tqdm(range(X.shape[0])):
        pred = torch.argmax(model( X[i].reshape(1, X.shape[1], X.shape[2]) ),dim=1).to('cpu')
        predictions.append(np.array([sequence[i].item(), pred.item()]))
    
    sol = np.array(predictions)
    
    out = pd.DataFrame(sol, columns=['sequence', 'state'])
    out = out.set_index('sequence')
    out.to_csv(f'./predictions_{no}.csv')

if __name__ == '__main__':

    full_dataset = Tab22_Dataset()
    train_len = int(len(full_dataset) * train_split)
    test_len = len(full_dataset) - train_len
    train_dataset, test_dataset = random_split(full_dataset, [train_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    #model = RNN(input_size, hidden_size, num_layers, num_classes)
    model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes)
    model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Train Network
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}')
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device).squeeze(1)
            targets = targets.to(device=device)

            # forward
            model = model.float()
            scores = model(data.float())
            loss = criterion(scores, targets.float())

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent update step/adam step
            optimizer.step()

        # Train accuracy
        num_correct = 0
        num_samples = 0

        # Set model to eval
        model.eval()

        with torch.no_grad():
            for x, y in tqdm(train_loader):
                x = x.to(device=device).squeeze(1)
                y = y.to(device=device)

                scores = model(x.float())
                predictions = scores.argmax(1)
                num_correct += (predictions == y.argmax(1)).sum()
                num_samples += predictions.size(0)

        # Toggle model back to train
        model.train()
        print(f'Train accuracy: {num_correct / num_samples}')


    #Check accuracy
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x.float())
            predictions = scores.argmax(1)
            num_correct += (predictions == y.argmax(1)).sum()
            num_samples += predictions.size(0)

    print(f'Test accuracy: {num_correct / num_samples}')

    if submit:
        write_submission(model, no)

    # Toggle model back to train
    model.train()