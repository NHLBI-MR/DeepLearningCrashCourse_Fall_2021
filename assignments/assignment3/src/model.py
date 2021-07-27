##################################################
## Deep learning crash course, assignment 3
##################################################
## Description : the CNN model, using pytorch
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

class BlockMobileNet(nn.Module):
    def __init__(self, kx, ky, input_channel, output_channel, name):
        """Define a block in small cnn, using the seperable convolution

        Args:
            kx (int): conv kernel size
            ky (int): conv kernel size
            output_channel (int): number of output channels
            name (str): a string name for this block; it is a good practice to name the module
        """
        super().__init__()

        # *** START CODE HERE ***
        # add layers, use the seperable convolution
        self.blocks = nn.Sequential()

        self.depth_conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=(kx, ky), stride=1, padding=(int(kx/2), int(ky/2)), groups=input_channel, bias=True)
        self.point_wise_conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=True)

        self.blocks.add_module(f"%s_depth_conv_1" % name, self.depth_conv1)
        self.blocks.add_module(f"%s_1x1_conv_1" % name, self.point_wise_conv1)
        self.blocks.add_module(f"%s_bn_1" % name, nn.BatchNorm2d(output_channel))
        self.blocks.add_module(f"%s_relu_1" % name, nn.ReLU())

        self.depth_conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=(kx, ky), stride=1, padding=(int(kx/2), int(ky/2)), groups=output_channel, bias=True)
        self.point_wise_conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=True)

        self.blocks.add_module(f"%s_depth_conv_2" % name, self.depth_conv2)
        self.blocks.add_module(f"%s_1x1_conv_2" % name, self.point_wise_conv2)
        self.blocks.add_module(f"%s_bn_2" % name, nn.BatchNorm2d(output_channel))
        self.blocks.add_module(f"%s_relu_2" % name, nn.ReLU())

        # initialize the conv weights
        nn.init.kaiming_normal_(self.blocks[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.blocks[1].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.blocks[4].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.blocks[5].weight, mode='fan_out', nonlinearity='relu')

        # *** END CODE HERE ***

    def forward(self, x):
        # *** START CODE HERE ***
        return self.blocks(x)
        # *** END CODE HERE ***

class BlockDownSampleMobileNet(nn.Module):
    def __init__(self, kx, ky, input_channel, output_channel, name):
        """Define a downsample block in small cnn, using seperable convolution

        Args:
            kx (int): conv kernel size
            ky (int): conv kernel size
            output_channel (int): number of output channels
            name (str): a string name for this block; it is a good practice to name the module
        """
        super().__init__()

        # *** START CODE HERE ***
        # all layers
        self.blocks = nn.Sequential()

        self.depth_conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=(kx, ky), stride=2, padding=(int(kx/2), int(ky/2)), groups=input_channel, bias=True)
        self.point_wise_conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=True)

        self.blocks.add_module(f"%s_depth_conv_1" % name, self.depth_conv1)
        self.blocks.add_module(f"%s_1x1_conv_1" % name, self.point_wise_conv1)
        self.blocks.add_module(f"%s_bn_1" % name, nn.BatchNorm2d(output_channel))
        self.blocks.add_module(f"%s_relu_1" % name, nn.ReLU())

        self.depth_conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=(kx, ky), stride=1, padding=(int(kx/2), int(ky/2)), groups=output_channel, bias=True)
        self.point_wise_conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1, padding=0, bias=True)

        self.blocks.add_module(f"%s_depth_conv_2" % name, self.depth_conv2)
        self.blocks.add_module(f"%s_1x1_conv_2" % name, self.point_wise_conv2)
        self.blocks.add_module(f"%s_bn_2" % name, nn.BatchNorm2d(output_channel))
        self.blocks.add_module(f"%s_relu_2" % name, nn.ReLU())

        # initialize the conv weights
        nn.init.kaiming_normal_(self.blocks[0].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.blocks[1].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.blocks[4].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.blocks[5].weight, mode='fan_out', nonlinearity='relu')

        # *** END CODE HERE ***

    def forward(self, x):
        # *** START CODE HERE ***
        return self.blocks(x)
        # *** END CODE HERE ***

class Cifar10SmallCNN(nn.Module):

    def __init__(self, H, W, C, use_mobile_net_conv=False):
        """Initial the model

        Please create the pytorch layers for the small CNN with the following architecture: 

        Im -> Block(5x5, 32) -> BlockDownSample(5x5, 64) -> Block(3x3, 64) -> BlockDownSample(3x3, 128) -> Block(3x3, 128) -> BlockDownSample(3x3, 256) -> Block(3x3, 256) -> flatten -> FC-256 -> BatchNorm1d -> ReLU -> FC-10

        or equivalent architecture using mobile-net blocks

        Args:
            H (int): Height of input image
            W (int): Width of input image
            C (int): Number of channels of input image
            use_mobile_net_conv (bool): If True, use the seperable convolution
        """
        super().__init__()

        # *** START CODE HERE ***
        # if use_mobile_net_conv is True, use the seperable convolution

        if(use_mobile_net_conv):
            self.b1 = BlockMobileNet(5, 5, C, 32, 'block_1')
            self.b2 = BlockDownSampleMobileNet(5, 5, 32, 64, 'blockdownsample_1')
            self.b3 = BlockMobileNet(3, 3, 64, 64, 'block_2')
            self.b4 = BlockDownSampleMobileNet(3, 3, 64, 128, 'blockdownsample_2')
            self.b5 = BlockMobileNet(3, 3, 128, 128, 'block_3')
            self.b6 = BlockDownSampleMobileNet(3, 3, 128, 256, 'blockdownsample_3')
            self.b7 = BlockMobileNet(3, 3, 256, 256, 'block_4')
        else:
            self.b1 = Block(5, 5, C, 32, 'block_1')
            self.b2 = BlockDownSample(5, 5, 32, 64, 'blockdownsample_1')
            self.b3 = Block(3, 3, 64, 64, 'block_2')
            self.b4 = BlockDownSample(3, 3, 64, 128, 'blockdownsample_2')
            self.b5 = Block(3, 3, 128, 128, 'block_3')
            self.b6 = BlockDownSample(3, 3, 128, 256, 'blockdownsample_3')
            self.b7 = Block(3, 3, 256, 256, 'block_4')

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
        x = self.b7(x)

        x = self.fc256(torch.flatten(x, 1))
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.output_fc10(x)

        return x
        # *** END CODE HERE ***

def main():

    # create the model
    m = Cifar10SmallCNN(32, 32, 3, use_mobile_net_conv=False)
    print(m)

    # test forward pass
    input = torch.rand((16, 3, 32, 32))
    output = m(input)
    print("output", output.shape)

    # use seperable conv
    m = Cifar10SmallCNN(32, 32, 3, use_mobile_net_conv=True)
    print(m)

    output = m(input)

if __name__ == '__main__':
    main()