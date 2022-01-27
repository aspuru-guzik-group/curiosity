import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T
import helper
import config








class StateDecoder(nn.Module):
    def __init__(self, n_state_neurons, output_size, device = 'cuda', sigmoid = False, noisy_net = False, unpack=False):
        super(StateDecoder, self).__init__()
        self.sigmoid = sigmoid
        self.device = device
        self.unpack = unpack
        
        self.fc1 = nn.Linear(n_state_neurons, output_size)

    
    def forward(self, x):
        if self.unpack:
            x, _ = x
        
        x = x.to(config.device)
        if self.sigmoid:
            return torch.softmax(self.fc1(x), dim=-1)
        else:
            return self.fc1(x)


class StateDecoderTwoHeads(nn.Module):
    def __init__(self, n_state_neurons, output_size_1, output_size_2,  device = 'cuda', sigmoid = False, noisy_net = False):
        super(StateDecoderTwoHeads, self).__init__()
        self.sigmoid = sigmoid
        self.device = device

        
        self.fc1_1 = nn.Linear(n_state_neurons, 128)
        self.bn1_1 = nn.BatchNorm1d(128)
        self.fc2_1 = nn.Linear(128, 64)
        self.bn2_1 = nn.BatchNorm1d(64)
        self.fc3_1 = nn.Linear(64, output_size_1)

        self.fc1_2 = nn.Linear(n_state_neurons, 128)
        self.bn1_2 = nn.BatchNorm1d(128)
        self.fc2_2 = nn.Linear(128, 64)
        self.bn2_2 = nn.BatchNorm1d(64)
        self.fc3_2 = nn.Linear(64, output_size_2)
    
    def forward(self, x):
        x = x.to(config.device)
        shape = x.shape


        x1 = F.relu(self.fc1_1(x))
        x1 = F.relu(self.fc2_1(x1))
        x1 = self.fc3_1(x1)

        x2 = F.relu(self.fc1_2(x))
        x2 = F.relu(self.fc2_2(x2))
        x2 = self.fc3_2(x2)
        
        return torch.softmax(x1, dim=-1), torch.softmax(x2, dim=-1)


class SELFIESEncoder(nn.Module):
    def __init__(self, n_state_neurons, device = 'cuda', num_layers = 1, dropout = 0, bidirectional = False):
        super(SELFIESEncoder, self).__init__()
        self.is_recurrent = True
        
        self.device = device
        self.num_layers = num_layers

        self.alphabet = helper.alphabet
        self.alphabet_size = helper.alphabet_size 

        if bidirectional:
            self.num_directions = 2
            self.n_state_neurons = n_state_neurons * 2
        else:
            self.num_directions = 1
            self.n_state_neurons = n_state_neurons

        
        self.lstm = nn.LSTM(self.alphabet_size, self.n_state_neurons, batch_first=True, num_layers=num_layers)

        self.reset_hidden_state()

        self.use_cell = False
        
    
    def forward(self, x):
        x = x.to(config.device)
        
        if self.use_cell:
            
            hx, cx = self.hidden_state[0]
            
            hx, cx = self.lstm_cells[0](x, (hx.to(config.device), cx.to(config.device)))
            hx, cx = self.lstm_cells[0](x, (hx, cx)) 
            
            self.hidden_state[0] = (hx, cx)
            
            
            for i in range(1, self.num_layers):
                hx_new, cx_new = self.hidden_state[i]
                
                hx, cx = self.lstm_cells[i](hx, (hx_new.to(config.device), cx_new.to(config.device)))
                hx, cx = self.lstm_cells[i](hx, (hx, cx)) 
                
                self.hidden_state[i] = (hx, cx)
                
            return hx
        else:
            output, _ = self.lstm(x)
            
            return output 

    def set_use_cell(self, use_cell):
        self.use_cell = use_cell

        
    def reset_hidden_state(self, batch_size = 1):
        self.hidden_state = []
        for i in range(self.num_layers):
            cx = torch.autograd.Variable(torch.zeros(batch_size, self.n_state_neurons))
            hx = torch.autograd.Variable(torch.zeros(batch_size, self.n_state_neurons))
            self.hidden_state.append((hx, cx))

    def truncate_hidden_state(self):

        for i in range(self.num_layers):
            hx, cx = self.hidden_state[i]
            hx, cx = hx[1:], cx[1:]
            self.hidden_state[i] = (hx, cx)

    def sync_cell_to_lstm(self):
        state_dict = self.lstm.state_dict()
        keys = []
        print(state_dict)
        for key in state_dict:
            new_key = key[:-3]
            keys.append((key, new_key))
        for key in keys:
            old_key, new_key = key
            state_dict[new_key] = state_dict.pop(old_key)
        self.lstm_cell.load_state_dict(state_dict)

    def sync_lstm_to_cell(self):
        state_dict = self.lstm_cell.state_dict()
        keys = []
        for key in state_dict:
            new_key = key + '_l0'
            keys.append((key, new_key))
        for key in keys:
            old_key, new_key = key
            state_dict[new_key] = state_dict.pop(old_key)
        self.lstm.load_state_dict(state_dict)





class SELFIESTransformerEncoder(nn.Module):
    def __init__(self, n_state_neurons, device = 'cuda', num_layers = 1, dropout = 0, bidirectional = False):
        super(SELFIESTransformerEncoder, self).__init__()
        self.is_recurrent = True
        
        self.device = device
        self.num_layers = num_layers

        self.alphabet = helper.alphabet
        self.alphabet_size = helper.alphabet_size 

        if bidirectional:
            self.num_directions = 2
            self.n_state_neurons = n_state_neurons * 2
        else:
            self.num_directions = 1
            self.n_state_neurons = n_state_neurons

        
        if num_layers == 1:
            self.lstm_cells = [nn.LSTMCell(self.alphabet_size, self.n_state_neurons).to(config.device)]
        else:
            self.lstm_cells = []
            self.lstm_cells.append(nn.LSTMCell(self.alphabet_size, self.n_state_neurons).to(config.device))
            for _ in range(num_layers-1):
                self.lstm_cells.append(nn.LSTMCell(self.n_state_neurons, self.n_state_neurons).to(config.device))

        self.lstm = nn.LSTM(self.alphabet_size, self.n_state_neurons, batch_first=True, num_layers=num_layers)

        self.reset_hidden_state()

        self.use_cell = True
        
    
    def forward(self, x):
        x = x.to(config.device)
        
        if self.use_cell:
            
            hx, cx = self.hidden_state[0]
            
            hx, cx = self.lstm_cells[0](x, (hx.to(config.device), cx.to(config.device)))
            hx, cx = self.lstm_cells[0](x, (hx, cx)) 
            
            self.hidden_state[0] = (hx, cx)
            
            
            for i in range(1, self.num_layers):
                hx_new, cx_new = self.hidden_state[i]
                
                hx, cx = self.lstm_cells[i](hx, (hx_new.to(config.device), cx_new.to(config.device)))
                hx, cx = self.lstm_cells[i](hx, (hx, cx)) 
                
                self.hidden_state[i] = (hx, cx)
                
            return hx
        else:
            output, _ = self.lstm(x)
            
            return output # (batch_size, seq_length, n_dims)


def get_pred_network(device, batch_size):

    state_embedding_prediction_module = nn.Linear(len(helper.alphabet), 10).to(device)
    state_encoder_prediction_module = nn.LSTM(10, 64, batch_first=True, bidirectional=False, dropout=0, num_layers=1).to(device)
    state_decoder_prediction_module = StateDecoder(64, 1, device = device, unpack=True).to(device)
    prediction_network = torch.nn.Sequential(state_embedding_prediction_module, state_encoder_prediction_module, state_decoder_prediction_module).to(device)

    return prediction_network