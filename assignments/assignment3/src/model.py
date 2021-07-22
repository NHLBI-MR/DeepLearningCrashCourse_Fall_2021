##################################################
## Deep learning crash course, assignment 2
##################################################
## Description : the N-layer MLP model, using pytorch
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Mmaintainer: xueh2@github
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

class Block(nn.Module):
    def __init__(self, kx, ky, input_channel, output_channel, name):
        """Define a block in small cnn

        Args:
            kx (int): conv kernel size
            ky (int): conv kernel size
            output_channel (int): number of output channels
            name (str): a string name for this block; it is a good practice to name the module
        """
        super().__init__()
               
        # *** START CODE HERE ***
        # add layers for a block
        self.blocks = nn.Sequential()
        
        self.blocks.add_module(f"%s_conv_1" % name, nn.Conv2d(input_channel, output_channel, kernel_size=(kx, ky), stride=1, padding=(int(kx/2), int(ky/2)), bias=True))
        self.blocks.add_module(f"%s_bn_1" % name, nn.BatchNorm2d(output_channel))
        self.blocks.add_module(f"%s_relu_1" % name, nn.ReLU())
        
        self.blocks.add_module(f"%s_conv_2" % name, nn.Conv2d(output_channel, output_channel, kernel_size=(kx, ky), stride=1, padding=(int(kx/2), int(ky/2)), bias=True))
        self.blocks.add_module(f"%s_bn_2" % name, nn.BatchNorm2d(output_channel))
        self.blocks.add_module(f"%s_relu_2" % name, nn.ReLU())
             
        # initialize the conv weights
        nn.init.kaiming_normal_(self.blocks[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.blocks[3].weight, mode='fan_out', nonlinearity='relu')
        
        # *** END CODE HERE ***
        
    def forward(self, x):
        # *** START CODE HERE ***
        return self.blocks(x)
        # *** END CODE HERE ***

class BlockDownSample(nn.Module):
    def __init__(self, kx, ky, input_channel, output_channel, name):
        """Define a downsample block in small cnn

        Args:
            kx (int): conv kernel size
            ky (int): conv kernel size
            output_channel (int): number of output channels
            name (str): a string name for this block; it is a good practice to name the module
        """
        super().__init__()
               
        # *** START CODE HERE ***
        # all layers for a block
        self.blocks = nn.Sequential()
        
        self.blocks.add_module(f"%s_conv_1" % name, nn.Conv2d(input_channel, output_channel, kernel_size=(kx, ky), stride=2, padding=(int(kx/2), int(ky/2)), bias=True))
        self.blocks.add_module(f"%s_bn_1" % name, nn.BatchNorm2d(output_channel))
        self.blocks.add_module(f"%s_relu_1" % name, nn.ReLU())
        
        self.blocks.add_module(f"%s_conv_2" % name, nn.Conv2d(output_channel, output_channel, kernel_size=(kx, ky), stride=1, padding=(int(kx/2), int(ky/2)), bias=True))
        self.blocks.add_module(f"%s_bn_2" % name, nn.BatchNorm2d(output_channel))
        self.blocks.add_module(f"%s_relu_2" % name, nn.ReLU())
        
        # initialize the conv weights
        nn.init.kaiming_normal_(self.blocks[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.blocks[3].weight, mode='fan_out', nonlinearity='relu')
        
        # *** END CODE HERE ***
        
    def forward(self, x):
        # *** START CODE HERE ***
        return self.blocks(x)
        # *** END CODE HERE ***
        
class Cifar10SmallCNN(nn.Module):

    def __init__(self, H, W, C):
        """Initial the model

        Please create the pytorch layers for the small CNN wit the following architecture: 
        
        Im -> Block(5x5, 32) -> BlockDownSample(5x5, 64) -> Block(3x3, 64) -> BlockDownSample(3x3, 128) -> Block(3x3, 128) -> BlockDownSample(3x3, 256) -> flatten -> FC-256 -> ReLU -> FC-10

        Args:
            H (int): Height of input image
            W (int): Width of input image
            C (int): Number of channels of input image
        """
        super().__init__()
        
        # *** START CODE HERE ***
        self.b1 = Block(5, 5, C, 32, 'block_1')
        self.b2 = BlockDownSample(5, 5, 32, 64, 'blockdownsample_1')
        self.b3 = Block(3, 3, 64, 64, 'block_2')
        self.b4 = BlockDownSample(3, 3, 64, 128, 'blockdownsample_2')
        self.b5 = Block(3, 3, 128, 128, 'block_3')
        self.b6 = BlockDownSample(3, 3, 128, 256, 'blockdownsample_3')
    
        input_dim = int(H/8) * int(W/8) * 256
                
        self.fc256 = nn.Linear(input_dim, 256, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.relu1 = nn.ReLU()
        self.output_fc10 = nn.Linear(256, 10, bias=True)
        
        # initialize the fc layer weights
        nn.init.xavier_normal_(self.fc256.weight)
        nn.init.xavier_normal_(self.output_fc10.weight)        
        # *** END CODE HERE ***

    def forward(self, x):
        """Forward pass of MLP model

        Args:
            x ([B, C, H, W]): a batch of input image

        Returns:
            output ([B, 10]): logits tensor, ready for the softmax
        """
        # *** START CODE HERE ***
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        
        x = self.fc256(torch.flatten(x, 1))
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.output_fc10(x)
        
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
    m = Cifar10SmallCNN(32, 32, 3)
    print(m)
    
    # test forward pass
    input = torch.rand((16, 3, 32, 32))
    output = m(input)
    print("output", output.shape)
    
if __name__ == '__main__':
    main()