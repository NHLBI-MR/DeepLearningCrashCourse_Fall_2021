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
    def __init__(self, kx, ky, channel, name):
        """Define a resnet block in unet
        Note: -->BN->ReLU->CONV->BN->ReLU->CONV--> + --> ...
               |-----------------------------------|
        Args:
            kx (int): conv kernel size
            ky (int): conv kernel size
            channel (int): number of channels
            name (str): a string name for this block; it is a good practice to name the module
        """
        super().__init__()

        # *** START CODE HERE ***
        # add layers for a block
        self.blocks = nn.Sequential()

        self.blocks.add_module(f"%s_bn_1" % name, nn.BatchNorm2d(channel))
        self.blocks.add_module(f"%s_relu_1" % name, nn.ReLU())
        self.blocks.add_module(f"%s_conv_1" % name, nn.Conv2d(channel, channel, kernel_size=(kx, ky), stride=1, padding=(int(kx/2), int(ky/2)), bias=True))

        self.blocks.add_module(f"%s_bn_2" % name, nn.BatchNorm2d(channel))
        self.blocks.add_module(f"%s_relu_2" % name, nn.ReLU())
        self.blocks.add_module(f"%s_conv_2" % name, nn.Conv2d(channel, channel, kernel_size=(kx, ky), stride=1, padding=(int(kx/2), int(ky/2)), bias=True))

        # initialize the conv weights
        nn.init.kaiming_normal_(self.blocks[2].weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.blocks[5].weight, mode='fan_out', nonlinearity='relu')

        # *** END CODE HERE ***

    def forward(self, x):
        # *** START CODE HERE ***
        return x + self.blocks(x)
        # *** END CODE HERE ***

class DownSample(nn.Module):
    def __init__(self, kx, ky, input_channel, output_channel, name):
        """Define a downsample block in unet

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
        self.blocks.add_module(f"%s_bn_1" % name, nn.BatchNorm2d(input_channel))
        self.blocks.add_module(f"%s_relu_1" % name, nn.ReLU())
        self.blocks.add_module(f"%s_conv_1" % name, nn.Conv2d(input_channel, output_channel, kernel_size=(kx, ky), stride=2, padding=(int(kx/2), int(ky/2)), bias=True))

        # initialize the conv weights
        nn.init.kaiming_normal_(self.blocks[2].weight, mode='fan_out', nonlinearity='relu')

        # *** END CODE HERE ***

    def forward(self, x):
        # *** START CODE HERE ***
        x = self.blocks(x)
        return x
        # *** END CODE HERE ***

class UpSample(nn.Module):
    def __init__(self, name, scale_factor=2.0):
        """Define a upsample block in unet, using torch.nn.functional.interpolate or torch.nn.Upsample

        Args:
            name (str): a string name for this block; it is a good practice to name the module
        """
        super().__init__()

        # *** START CODE HERE ***
        self.scale_factor = scale_factor
        # *** END CODE HERE ***

    def forward(self, x):
        # *** START CODE HERE ***
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        return x
        # *** END CODE HERE ***

class Concetenate(nn.Module):
    def __init__(self, kx, ky, input_channel1, input_channel2, output_channel, name):
        """Define a concetenation block in unet
            

        Args:
            kx (int): conv kernel size
            ky (int): conv kernel size
            input_channel1 (int): number of input channels
            input_channel2 (int): number of input channels
            output_channel (int): number of output channels
            name (str): a string name for this block; it is a good practice to name the module
        """
        super().__init__()

        # *** START CODE HERE ***
        self.blocks = nn.Sequential()
        self.blocks.add_module(f"%s_bn_1" % name, nn.BatchNorm2d(input_channel1+input_channel2))
        self.blocks.add_module(f"%s_relu_1" % name, nn.ReLU())
        self.blocks.add_module(f"%s_conv_1" % name, nn.Conv2d(input_channel1+input_channel2, output_channel, kernel_size=(kx, ky), stride=1, padding=(int(kx/2), int(ky/2)), bias=True))

        # initialize the conv weights
        nn.init.kaiming_normal_(self.blocks[2].weight, mode='fan_out', nonlinearity='relu')
        # *** END CODE HERE ***

    def forward(self, x1, x2):
        """forward pass for concetenate layer

        Args:
            x1 ([B, C1, H, W]) : first input
            x2 ([B, C2, H, W]) : second input

        Returns:
            x ([B, output_channel, H', W']): output
        """
        # *** START CODE HERE ***
        x = torch.cat([x1, x2], dim=1)
        x = self.blocks(x)
        return x
        # *** END CODE HERE ***

class UNet(nn.Module):

    def __init__(self, H, W, C):
        """Initial the model

        Please create the pytorch layers for the unet CNN, as planned in the assignment 3.

        Args:
            H (int): Height of input image
            W (int): Width of input image
            C (int): Number of channels of input image
        """
        super().__init__()

        # *** START CODE HERE ***

        self.input_conv = nn.Conv2d(C, 32, kernel_size=(5, 5), stride=1, padding=(2, 2), bias=True)

        self.down_sample_1 = DownSample(3, 3, 32, 64, 'down_sample_1')
        self.down_b1 = ResBlock(3, 3, 64, 'down_block_1')

        self.down_sample_2 = DownSample(3, 3, 64, 128, 'down_sample_2')
        self.down_b2 = ResBlock(3, 3, 128, 'down_block_2')

        self.down_sample_3 = DownSample(3, 3, 128, 128, 'down_sample_3')
        self.down_b3 = ResBlock(3, 3, 128, 'down_block_3')

        self.up_sample_1 = UpSample('up_sample_1')
        self.up_concetenate1 = Concetenate(3, 3, 128, 128, 128, 'up_concetenate_1')
        self.up_b1 = ResBlock(3, 3, 128, 'up_block_1')

        self.up_sample_2 = UpSample('up_sample_2')
        self.up_concetenate2 = Concetenate(3, 3, 64, 128, 64, 'up_concetenate_2')
        self.up_b2 = ResBlock(3, 3, 64, 'up_block_2')

        self.up_sample_3 = UpSample('up_sample_3')
        self.up_concetenate3 = Concetenate(3, 3, 32, 64, 32, 'up_concetenate_3')

        self.output_conv = nn.Conv2d(32, 1, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=True)

        # initialize the conv layer weights
        nn.init.kaiming_normal_(self.input_conv.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.output_conv.weight, mode='fan_out')

        # *** END CODE HERE ***

    def forward(self, x):
        """Forward pass of Unet model

        Args:
            x ([B, C, H, W]): a batch of input image

        Returns:
            output ([B, 1, H, W]): logits tensor
        """
        # *** START CODE HERE ***
        d1 = self.input_conv(x)

        d2 = self.down_sample_1(d1)
        d2 = self.down_b1(d2)

        d3 = self.down_sample_2(d2)
        d3 = self.down_b2(d3)

        d4 = self.down_sample_3(d3)
        d4 = self.down_b3(d4)

        u1 = self.up_sample_1(d4)
        u1 = self.up_concetenate1(d3, u1)
        u1 = self.up_b1(u1)

        u2 = self.up_sample_2(u1)
        u2 = self.up_concetenate2(d2, u2)
        u2 = self.up_b2(u2)

        u3 = self.up_sample_3(u2)
        u3 = self.up_concetenate3(d1, u3)

        output = self.output_conv(u3)

        return output
        # *** END CODE HERE ***

# --------------------------------------------------

class LossBinarySegmentation:
    """
    Loss for binary segmentation
    """

    def __init__(self):
        self.bce_loss = nn.BCELoss()

    def __call__(self, scores, y):
        """Compute binary CE loss

        Args:
            scores ([B, 1, H, W]): logits from the model, not the probability
            y ([B, 1, H, W]): mask

        Returns:
            loss (tensor): BCE loss
        """
        # *** START CODE HERE ***
        # TODO: compute and return the loss
        probs = torch.sigmoid(scores)
        loss = self.bce_loss(probs, y)
        # *** END CODE HERE ***
        return loss

# --------------------------------------------------

def main():

    # create the model
    m = UNet(112, 168, 3)
    print(m)

    # test forward pass
    input = torch.rand((16, 3, 112, 168), dtype=torch.float32)
    output = m(input)
    print("output", output.shape)

    y = torch.rand((16, 1, 112, 168), dtype=torch.float32)
    loss = LossBinarySegmentation()
    L = loss(output, y)
    print(L.item())

if __name__ == '__main__':
    main()