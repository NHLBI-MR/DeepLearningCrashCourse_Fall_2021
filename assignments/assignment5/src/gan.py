##################################################
## Deep learning crash course, assignment 5
##################################################
## Description : the gan model
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
from torch.nn import functional as F

import util

class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return torch.reshape(x, (x.size(0), *self.shape))
    
class Generator(nn.Module):

    def __init__(self, D):
        """The generator model

        Args:
            D (int): dimension of latent vector
        """
        super().__init__()
    
        # *** START CODE HERE ***

        self.input = torch.nn.Linear(D, 512)
        self.bn1 = torch.nn.BatchNorm1d(512)
        self.relu1 = torch.nn.ReLU(inplace=False)

        self.L2 = torch.nn.Linear(512, 64 * 7 * 7)
        self.bn2 = torch.nn.BatchNorm1d(64 * 7 * 7)
        self.relu2 = torch.nn.ReLU(inplace=False)
        self.reshape1 = Reshape(64, 7, 7)

        self.upsample1 = torch.nn.PixelShuffle(2)
        self.conv1 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.relu3 = torch.nn.ReLU(inplace=False)

        self.upsample2 = torch.nn.PixelShuffle(2)
        self.output = torch.nn.Conv2d(8, 1, kernel_size=3, padding=1)

        # initialize the conv layer weights
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.xavier_uniform_(self.L2.weight)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.output.weight, mode='fan_out')

        # *** END CODE HERE ***

    def forward(self, x):
        """Forward pass

        Args:
            x ([B, D]): the latent vectors

        Returns:
            output ([B, 1, H, W]): generated images
        """
        # *** START CODE HERE ***
        x = self.input(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.L2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.reshape1(x)

        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.upsample2(x)
        x = torch.tanh(self.output(x))

        return x
        # *** END CODE HERE ***
        
class Discriminator(nn.Module):
    
    def __init__(self, H, W, C):
        """The discriminator model

        Args:
            H (int): Height of input image
            W (int): Width of input image
            C (int): Number of channels of input image
        """
        super().__init__()

        # *** START CODE HERE ***

        self.input = torch.nn.Conv2d(C, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu1 = torch.nn.LeakyReLU(0.1, inplace=False)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.relu2 = torch.nn.LeakyReLU(0.1, inplace=False)

        N = 64 * int(H//4) * int(W//4)
        self.reshape1 = Reshape(N)
        self.linear1 = torch.nn.Linear(N, 512)
        self.bn3 = torch.nn.BatchNorm1d(512)
        self.relu3 = torch.nn.LeakyReLU(0.1, inplace=False)

        self.output = torch.nn.Linear(512, 1)
        
        # initialize the conv layer weights
        nn.init.kaiming_normal_(self.input.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out')
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.output.weight)

        # *** END CODE HERE ***

    def forward(self, x):
        """Forward pass

        Args:
            x ([B, 1, H, W]): a batch of input image

        Returns:
            output ([B, 1]): logits tensor
        """
        # *** START CODE HERE ***

        x = self.input(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.reshape1(x)
        x = self.linear1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.output(x)
               
        return x
        # *** END CODE HERE ***

# --------------------------------------------------

def loss_D(logit_real, logit_fake):
    """compute and return loss for discriminator

    Args:
        logit_real ([B, 1]): logits for the real samples
        logit_fake ([B, 1]): logits for the fake samples

    Returns:
        l_d: the discriminator loss
        
    Note : l_d is a pytorch loss. l_d.backward() will be called during backprop
    Pytorch has utility functions, such as torch.sigmoid or F.logsigmoid
    """
    
    # *** START CODE HERE ***
    l_d = F.logsigmoid(logit_real) + torch.log(1 - torch.sigmoid(logit_fake))
    l_d = torch.mean(-l_d, dim=0) 
    # *** END CODE HERE ***
    
    return l_d
    
def loss_G(logit_fake):
    """compute and return loss for generator

    Args:
        logit_fake ([B, 1]): logits for the fake samples

    Returns:
        l_g: the generator loss        
    """
    
    # *** START CODE HERE ***
    l_g = torch.mean(-F.logsigmoid(logit_fake), dim=0)
    # *** END CODE HERE ***
    
    return l_g

# --------------------------------------------------

def wgan_loss_D(logit_real, logit_fake, reg_s, lamda):
    """compute and return loss for discriminator, WGAN-GP

    Args:
        logit_real ([B, 1]): logits for the real samples
        logit_fake ([B, 1]): logits for the fake samples
        reg_s ([B, 1]): gradient penalty term
        lamda (float) : regularization strength
        
    Returns:
        l_d: the discriminator loss
        
    Note : l_d is a pytorch loss. l_d.backward() will be called during backprop
    """
    
    # *** START CODE HERE ***
    l_d = logit_fake - logit_real + lamda * reg_s
    l_d = torch.mean(l_d, dim=0)
    # *** END CODE HERE ***
    
    return l_d
    
def wgan_loss_G(logit_fake):
    """compute and return loss for generator, WGAN-GP

    Args:
        logit_fake ([B, 1]): logits for the fake samples

    Returns:
        l_g: the generator loss for wgan
    """
    
    # *** START CODE HERE ***
    l_g = -torch.mean(logit_fake, dim=0)
    # *** END CODE HERE ***
    
    return l_g

# --------------------------------------------------

def main():

    device = util.find_GPU()

    G = Generator(10).to(device=device)
    D = Discriminator(28,28,1).to(device=device)
    
    z = torch.randn(128, 10).to(device=device)
    
    fake_images = G(z)    
    
    # for the tesing purpose, we only need an array
    logit_real = D(z)
    logit_fake = D(z)

    l_d = loss_D(logit_real, logit_fake)
    l_g = loss_G(logit_fake)

if __name__ == '__main__':
    main()