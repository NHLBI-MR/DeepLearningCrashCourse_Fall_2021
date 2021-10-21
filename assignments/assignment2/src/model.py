##################################################
## Deep learning crash course, assignment 2
##################################################
## Description : the N-layer MLP model, using pytorch
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Maintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

import torch
import torch.nn as nn

import os
import sys
from pathlib import Path

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

class PytorchMLP(nn.Module):

    def __init__(self, H, W, C, num_hidden_layers):
        """Initial the model

        Please create the pytorch layers for MLP. Please use ReLU as the nonlinear activation. 
        Hints: torch.nn.Sequential may be useful. Also, check torch.nn.Linear and torch.nn.ReLU

        Args:
            H (int): Height of input image
            W (int): Width of input image
            C (int): Number of channels of input image
            num_hidden_layers (list, optional): number of hidden layers. Defaults to [300, 300, 200, 100].
        """
        super().__init__()
        
        # *** START CODE HERE ***
        self.blocks = nn.Sequential()
        for i, num_neurons in enumerate(num_hidden_layers):
            if(i==0):
                input_dim = int(H*W*C)
            else:
                input_dim = num_hidden_layers[i-1]
                
            output_dim = num_neurons
                
            self.blocks.add_module("fc_%d" % i, nn.Linear(input_dim, output_dim, bias=True))
            self.blocks.add_module("relu_%d" % i, nn.ReLU())
                
        self.blocks.add_module("fc_output", nn.Linear(output_dim, 10, bias=True))
        
        # *** END CODE HERE ***

    def forward(self, x):
        """Forward pass of MLP model

        Args:
            x ([B, C, H, W]): a batch of input image

        Returns:
            output ([B, 10]): logits tensor, ready for the softmax
        """
        # *** START CODE HERE ***
        x = torch.flatten(x, 1)
        x = self.blocks(x)
        return x
        # *** END CODE HERE ***        

def compute_test_accuracy(loader, model, loss_func, device=torch.device('cpu')):
    
    running_loss_train = 0.0
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            x, y = data            
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
                                          
            y_hat = model(x)
            loss = loss_func(y_hat, y)
            
            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            running_loss_train += loss.item()

        loss = running_loss_train / (i+1)
        accu = correct / total
    
    return loss, accu
              
def main():

    # create the model
    m = PytorchMLP(32, 32, 3, [300, 200, 100])
    print(m)
    
    # test forward pass
    input = torch.rand((16, 3, 32, 32))
    output = m(input)
    print("output", output.shape)
    
if __name__ == '__main__':
    main()