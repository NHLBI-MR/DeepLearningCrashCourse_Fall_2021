
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def find_GPU():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('using device:', device)

    return device

def one_hot_encode(arr, n_labels):
    """Perform one-hot encoding
    
    Borrowed from https://www.kaggle.com/francescapaulin/character-level-lstm-in-pytorch
    """
    
    # Initialize the the encoded array
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)
    
    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.
    
    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot

def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
       
       Borrowed from https://www.kaggle.com/francescapaulin/character-level-lstm-in-pytorch
    '''
    
    batch_size_total = batch_size * seq_length
    # total number of batches we can make, // integer division, round down
    n_batches = len(arr)//batch_size_total
    
    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows, n. of first row is the batch size, the other lenght is inferred
    arr = arr.reshape((batch_size, -1))
    
    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n+seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y
        
def predict(net, char, h=None, top_k=None, device=torch.device('cpu')):
    ''' Given a character, predict the next character.
        Returns the predicted character and the hidden state.
        
        Borrowed from https://www.kaggle.com/francescapaulin/character-level-lstm-in-pytorch
    '''
    
    # tensor inputs
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)
    
    inputs = inputs.to(device=device)
    net.to(device=device)
    
    # detach hidden state from history
    if(type(h) is tuple):
        h = tuple([each.data for each in h])
    else:
        h = h.data
        
    # get the output of the model
    out, h = net(inputs, h)

    # get the character probabilities
    # apply softmax to get p probabilities for the likely next character giving x
    p = F.softmax(out, dim=1).data
    p = p.cpu() # move to cpu
    
    # get top characters
    # considering the k most probable characters with topk method
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()
    
    # select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())
    
    # return the encoded value of the predicted char and the hidden state
    return net.int2char[char], h
              
def sample(net, size, prime='Il', top_k=None, device=torch.device('cpu')):
    """Sample the model by doing the self-regression
    
    Borrowed from https://www.kaggle.com/francescapaulin/character-level-lstm-in-pytorch
    """    
    
    net.eval() # eval mode
    net.to(device=device)
    
    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k, device=device)

    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k, device=device)
        chars.append(char)

    return ''.join(chars)

def plot_ecg_waves(waves, triggers, names, figsize=[32, 32]):
    """plot ecg waves as a plane with trigger

    Args:
        waves ([B, T, C]): C channels of ecg waves to plot
        triggers ([B, T]): Triggers to be plotted
        names (list): names of case
        figsize (list, optional): figure size. Defaults to [32, 32].

    Returns:
        figs : handles to figure
    """
    
    B, T, C = waves.shape
    
    figs = []
    
    for b in range(B):
        ind = np.argwhere(triggers[b, :]==1.0)
        
        fig=plt.figure(figsize=figsize)    
        for c in range(C):
            fig.add_subplot(C, 1, c+1)
            plt.plot(waves[b, :, c].squeeze(), 'k');
            plt.plot(ind, waves[b, ind, c], 'ro');
                
        plt.title(names[b])
        figs.append(fig)
    
    return figs