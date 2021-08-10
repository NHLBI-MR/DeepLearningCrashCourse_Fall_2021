##################################################
## Deep learning crash course, assignment 4
##################################################
## Description : the RNN model
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Maintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

import os
import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

import torch
import torch.nn as nn

# ----------------------------------------------------------

class char_rnn_lstm(nn.Module):
    
    def __init__(self, tokens, n_hidden=512, n_layers=3, drop_prob=0.5):
        """Declear the char RNN with LSTM

        Args:
            tokens : tokens for training
            n_hidden (int, optional): dimension of internal state of RNN. Defaults to 512.
            n_layers (int, optional): number of layers of RNN. Defaults to 3.
            drop_prob (float, optional): probability of dropout. Defaults to 0.5.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.n_tokens = len(self.chars)
                
        # *** START CODE HERE ***
        # define a simple RNN
        # x -> LSTM -> drop_out -> Linear layer
        # note batch is the first dimension
        # use drop out in LSTM
        # pls. check the LSTM at https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM
        
        self.lstm = nn.LSTM(self.n_tokens, n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, self.n_tokens)
        # *** END CODE HERE ***
      
    def forward(self, x, hidden):
        """Forward pass
        
        Args:
        
            x ([B, seq_length, n_tokens]) : one-hot-encoded batch, order of dimension is batch, time, token
            hidden (tuple of two tensors) : hidden state and cell state of LSTM, [n_layers, n_batch_size, n_hidden]
            
        Outputs: 
            out ([B*seq_length, n_tokens]) : output logits
            hidden (tuple of two tensors [n_layers, batch_size, n_hidden]) : updated hidden and cell states
            
        Note: the out is 2D tensor to call CE loss; you may want to use torch.reshape
        """
                
        # *** START CODE HERE ***
        
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)        
        out = self.fc(out)
        out  = torch.reshape(out, (-1, self.n_tokens))
        
        # *** END CODE HERE ***
        
        return out, hidden
    
    
    def init_hidden(self, batch_size, device):
        ''' Initializes hidden and cell state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        hidden = (torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device),
                   torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device))
        
        return hidden

# ----------------------------------------------------------
    
class char_rnn_gru(nn.Module):
    
    def __init__(self, tokens, n_hidden=512, n_layers=3, drop_prob=0.5):
        """Declear the char RNN with GRU

        Args:
            tokens: tokens used for training
            n_hidden (int, optional): dimension of internal state of RNN. Defaults to 512.
            n_layers (int, optional): number of layers of RNN. Defaults to 3.
            drop_prob (float, optional): probability of dropout. Defaults to 0.5.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.n_tokens = len(self.chars)
                
        # *** START CODE HERE ***
        # define a simple GRU
        # x -> GRU -> drop_out -> Linear layer
        # note batch is the first dimension
        # use drop out in GRU
        # pls. check the GRU at https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU
        
        self.gru = nn.GRU(self.n_tokens, n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden, self.n_tokens)
        # *** END CODE HERE ***
      
    def forward(self, x, hidden):
        """Forward pass
        
        Args:
        
            x ([B, seq_length, n_tokens]) : one-hot-encoded batch, order of dimension is batch, time, token
            hidden (tuple of two tensors) : hidden state and cell state of LSTM, [n_layers, n_batch_size, n_hidden]
            
        Outputs: 
            out ([B*seq_length, n_tokens]) : output logits
            hidden (tuple of two tensors [n_layers, batch_size, n_hidden]) : updated hidden and cell states
            
        Note: the out is 2D tensor to call CE loss; you may want to use torch.reshape
        """
                
        # *** START CODE HERE ***
        
        r_output, hidden = self.gru(x, hidden)
        out = self.dropout(r_output)        
        out = self.fc(out)
        out  = torch.reshape(out, (-1, self.n_tokens))
        
        # *** END CODE HERE ***
        
        return out, hidden
    
    
    def init_hidden(self, batch_size, device):
        ''' Initializes hidden state '''
        hidden = torch.zeros(self.n_layers, batch_size, self.n_hidden, device=device)
        return hidden
    
# ----------------------------------------------------------

def main():
    """Model testing code
    """
    pass

if __name__ == '__main__':
    main()