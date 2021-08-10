##################################################
## Deep learning crash course, assignment 4
##################################################
## Description : the transformer model
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Maintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

import os
import sys
import math
import logging
from pathlib import Path

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

# ----------------------------------------------------------           

# This implemenation of attention and transformers is adopted from https://github.com/karpathy/minGPT

# ----------------------------------------------------------           
class SelfAttention(nn.Module):
    """
    Multi-head attention model    
    Dropout is added on the attention matrix and output.    
    """

    def __init__(self, C=4, T=1024, output_size=1, is_causal=False, n_embd=128, n_head=8, dropout_p=0.1):
        """Define the layers for a self-attention

            Input to the attention layer has the size [B, T, C]
            Output has the size [B, T, output_size]
            
            Internally, the attention has embedding size n_embd

        Args:
            C (int, optional): input dimension [B, T, C]. Defaults to 4.
            T (int, optional): number of time points for attention layer. Defaults to 1024.
            output_size (int, optional): number of output dimension. Defaults to 1.
            is_causal (bool, optional): whether applying the masking to make the layer causal. Defaults to False.
            n_embd (int, optional): number of internal dimension. Defaults to 128.
            n_head (int, optional): number of heads. Defaults to 8.
        """
        super().__init__()            
        
        self.C = C
        self.T = T
        self.output_size = output_size
        self.is_causal = is_causal
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout_p = dropout_p
        
        # key, query, value projections matrix
        # Wk, Wq, Wv
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
                
        self.output_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout_p)
        self.resid_drop = nn.Dropout(dropout_p)
    
        self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))
                
    def forward(self, x):
        """forward pass for the 

        Args:
            x ([B, T, C]): Input of a batch of time series

        Returns:
            y: logits in the shape of [B, T, output_size]
        """
        
        B, T, C = x.size()

        # apply the key, query and value matrix
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Compute attention matrix, use the matrix broadcasing 
        # https://pytorch.org/docs/stable/notes/broadcasting.html
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # if causality is needed, apply the mask
        if(self.is_causal):
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.output_proj(y))
        return y

class Block(nn.Module):
    """ Transformer module
    
    The Pre-LayerNorm implementation is used here:
    
    x-> LayerNorm -> attention -> + -> LayerNorm -> LinearLayers -> + -> logits
    |-----------------------------| |-------------------------------|
    
    """

    def __init__(self, C=4, T=1024, output_size=1, is_causal=False, n_embd=128, n_head=8, dropout_p=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(C=C, T=T, output_size=output_size, is_causal=is_causal, n_embd=n_embd, n_head=n_head, dropout_p=dropout_p)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout_p),
        )

    def forward(self, x):        
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class ECGDetector(nn.Module):
    """A transformer based ECG detector

        This model uses positional embedding. 
        
        The architecture is quite straight-forward :
        
        x -> input_proj --> + --> drop_out --> attention layers one after another --> LayerNorm --> output_proj --> logits
                            |
        pos_emb ------------|
    
    """

    def __init__(self, n_layer=8, C=4, T=1024, output_size=1, is_causal=False, n_embd=128, n_head=8, dropout_p=0.1):
        super().__init__()

        self.T = T

        # input projection
        self.input_proj = nn.Linear(C, n_embd)
        
        # the positional embedding is used
        # this is learned through the training
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_embd))
                
        self.drop = nn.Dropout(dropout_p)
        
        # transformer modules
        # stack them for n_layers
        self.blocks = nn.Sequential(*[Block(C=C, T=T, output_size=output_size, is_causal=is_causal, n_embd=n_embd, n_head=n_head, dropout_p=dropout_p) for _ in range(n_layer)])
        
        # decoder head
        self.layer_norm = nn.LayerNorm(n_embd)
        self.output_proj = nn.Linear(n_embd, output_size, bias=False)

        self.apply(self._init_weights)

        # a good trick to count how many parameters
        logger.info("number of parameters: %d", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """Forward pass of detector

        Args:
            x ([B, T, C]]): Input time sereis with B batches and T time points with C channels

            Due to the positional embedding is used, the input T is limited to less or equal to self.T

        Returns:
            logits: [B, T, output_size]
        """
        
        B, T, C = x.size()
        assert T <= self.T, "The positional embedding is used, so the maximal series length is %d" % self.T
                
        # *** START CODE HERE ***
        # Todo : finish the forward pass
        
        # project input from C channels to n_embd channels
        x_proj = self.input_proj(x)
        x = self.drop(x_proj + self.pos_emb[:, :T, :])
        
        # go through all layers of attentions
        x = self.blocks(x)
        
        # project outputs to output_size channel        
        x = self.layer_norm(x)
        logits = self.output_proj(x)
        # *** END CODE HERE ***
        
        return logits
    
# -------------------------------------------

class LossTriggerDetection:
    """
    Loss for trigger detection
    """

    def __init__(self):
        self.bce_loss = nn.BCELoss()

    def __call__(self, scores, y):
        """Compute binary CE loss

        Args:
            scores ([B, T, 1]): logits from the model, not the probability
            y ([B, T]): probability whether a time point is a trigger

        Returns:
            loss (tensor): BCE loss
        """
        # *** START CODE HERE ***
        # TODO: compute and return the loss
        probs = torch.sigmoid(scores)
        loss = self.bce_loss(probs.squeeze(), y)
        # *** END CODE HERE ***
        return loss
                
# ----------------------------------------------------------

def main():
    """Model testing code
    """
    pass

if __name__ == '__main__':
    main()