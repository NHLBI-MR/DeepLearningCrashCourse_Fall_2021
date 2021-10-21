##################################################
## Deep learning crash course, assignment 3
##################################################
## Description : the resnet model, using pytorch
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

class ResBlock(nn.Module):
    def __init__(self, kx, ky, input_channel, output_channel, name):
        """Define a block in small resnet
        Note: ... BN->ReLU->CONV ...

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

        self.blocks.add_module(f"%s_bn_1" % name, nn.BatchNorm2d(input_channel))
        self.blocks.add_module(f"%s_relu_1" % name, nn.ReLU())
        self.blocks.add_module(f"%s_conv_1" % name, nn.Conv2d(input_channel, output_channel, kernel_size=(kx, ky), stride=1, padding=(int(kx/2), int(ky/2)), bias=True))

        self.blocks.add_module(f"%s_bn_2" % name, nn.BatchNorm2d(output_channel))
        self.blocks.add_module(f"%s_relu_2" % name, nn.ReLU())
        self.blocks.add_module(f"%s_conv_2" % name, nn.Conv2d(output_channel, output_channel, kernel_size=(kx, ky), stride=1, padding=(int(kx/2), int(ky/2)), bias=True))

        # initialize the conv weights
        nn.init.kaiming_normal_(self.blocks[2].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.blocks[5].weight, mode='fan_out', nonlinearity='relu')

        # *** END CODE HERE ***

    def forward(self, x):
        # *** START CODE HERE ***
        return x + self.blocks(x)
        # *** END CODE HERE ***

class ResBlockDownSample(nn.Module):
    def __init__(self, kx, ky, input_channel, output_channel, name):
        """Define a downsample block in small resnet

        Args:
            kx (int): conv kernel size
            ky (int): conv kernel size
            output_channel (int): number of output channels
            name (str): a string name for this block; it is a good practice to name the module
        """
        super().__init__()

        # *** START CODE HERE ***
        # all layers for a block
        self.blocks_1 = nn.Sequential()
        self.blocks_1.add_module(f"%s_bn_1" % name, nn.BatchNorm2d(input_channel))
        self.blocks_1.add_module(f"%s_relu_1" % name, nn.ReLU())
        self.blocks_1.add_module(f"%s_conv_1" % name, nn.Conv2d(input_channel, output_channel, kernel_size=(kx, ky), stride=2, padding=(int(kx/2), int(ky/2)), bias=True))

        self.blocks_2 = nn.Sequential()
        self.blocks_2.add_module(f"%s_bn_2" % name, nn.BatchNorm2d(output_channel))
        self.blocks_2.add_module(f"%s_relu_2" % name, nn.ReLU())
        self.blocks_2.add_module(f"%s_conv_2" % name, nn.Conv2d(output_channel, output_channel, kernel_size=(kx, ky), stride=1, padding=(int(kx/2), int(ky/2)), bias=True))

        # initialize the conv weights
        nn.init.kaiming_normal_(self.blocks_1[2].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.blocks_2[2].weight, mode='fan_out', nonlinearity='relu')

        # *** END CODE HERE ***

    def forward(self, x):
        # *** START CODE HERE ***
        x1 = self.blocks_1(x)
        x2 = self.blocks_2(x1)
        return x1 + x2
        # *** END CODE HERE ***

class Cifar10SmallResNet(nn.Module):

    def __init__(self, H, W, C):
        """Initial the model

        Please create the pytorch layers for the small CNN with the following architecture: 

        Im -> CONV(5x5, 32) -> ResBlock(3x3, 32) -> ResBlock(3x3, 32) -> ResBlockDownSample(3x3, 64) -> ResBlock(3x3, 64) -> ResBlock(3x3, 64) -> ResBlockDownSample(3x3, 128) -> ResBlock(3x3, 128) -> ResBlock(3x3, 128) -> ResBlockDownSample(3x3, 256) -> BatchNorm2d -> ReLU -> CONV(1x1, 512) -> BatchNorm2d -> ReLU -> CONV(?x?, 512) -> flatten -> FC-256 -> BatchNorm1d -> ReLU -> FC-10

        Args:
            H (int): Height of input image
            W (int): Width of input image
            C (int): Number of channels of input image
        """
        super().__init__()

        # *** START CODE HERE ***

        self.input_conv = nn.Conv2d(C, 32, kernel_size=(5, 5), stride=1, padding=(2, 2), bias=True)

        self.b1 = ResBlock(3, 3, 32, 32, 'block_1')
        self.b2 = ResBlock(3, 3, 32, 32, 'block_2')
        self.b3 = ResBlockDownSample(3, 3, 32, 64, 'blockdownsample_1')

        self.b4 = ResBlock(3, 3, 64, 64, 'block_3')
        self.b5 = ResBlock(3, 3, 64, 64, 'block_4')
        self.b6 = ResBlockDownSample(3, 3, 64, 128, 'blockdownsample_2')

        self.b7 = ResBlock(3, 3, 128, 128, 'block_5')
        self.b8 = ResBlock(3, 3, 128, 128, 'block_6')
        self.b9 = ResBlockDownSample(3, 3, 128, 256, 'blockdownsample_3')

        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.relu1 = nn.ReLU()
        self.b10 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=True)

        self.bn2 = nn.BatchNorm2d(num_features=512)
        self.relu2 = nn.ReLU()
        self.b11 = nn.Conv2d(512, 512, kernel_size=(4, 4), stride=1, padding=(0, 0), bias=True)

        input_dim = 512

        self.fc256 = nn.Linear(input_dim, 256, bias=True)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.relu3 = nn.ReLU()
        self.output_fc10 = nn.Linear(256, 10, bias=True)

        # initialize the conv layer weights
        nn.init.kaiming_normal_(self.input_conv.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.b10.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.b11.weight, mode='fan_out')

        # initialize the fc layer weights
        nn.init.xavier_normal_(self.fc256.weight)
        nn.init.xavier_normal_(self.output_fc10.weight)
        # *** END CODE HERE ***

    def forward(self, x):
        """Forward pass of resnet model

        Args:
            x ([B, C, H, W]): a batch of input image

        Returns:
            output ([B, 10]): logits tensor, ready for the softmax
        """
        # *** START CODE HERE ***
        x = self.input_conv(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        x = self.b7(x)
        x = self.b8(x)
        x = self.b9(x)

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.b10(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.b11(x)

        x = self.fc256(torch.flatten(x, 1))
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.output_fc10(x)

        return x
        # *** END CODE HERE ***

def main():

    # create the model
    m = Cifar10SmallResNet(32, 32, 3)
    print(m)

    # test forward pass
    input = torch.rand((16, 3, 32, 32))
    output = m(input)
    print("output", output.shape)

if __name__ == '__main__':
    main()