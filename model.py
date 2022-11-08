import torch
from torch import nn
import pdb

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(LSTMPredictor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.pre_fc = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        
    def forward(self, x, hidden):
        x = self.pre_fc(x)
        x = self.relu(x)
        
        lstm_out, hidden = self.lstm(x, hidden)

        lstm_out = self.relu(lstm_out)

        out = self.fc(lstm_out)

        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        
        return hidden

class RNNpredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(RNNpredictor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.pre_fc = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.RNN(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        
    def forward(self, x, hidden):
        x = self.pre_fc(x)
        x = self.relu(x)

        rnn_out, hidden = self.rnn(x, hidden[0])

        rnn_out = self.relu(rnn_out)

        out = self.fc(rnn_out)

        return out, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        
        return hidden