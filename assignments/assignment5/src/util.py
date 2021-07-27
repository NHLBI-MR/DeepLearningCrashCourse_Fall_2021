import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

def find_GPU():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('using device:', device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    return device

def dice_coeff(y_pred, y):
    a = y_pred.view(-1).float()
    b = y.view(-1).float()
    inter = torch.dot(a, b) + 0.0001
    union = torch.sum(a) + torch.sum(b) + 0.0001
    return 2*inter.float()/union.float()

def plot_image_array(im, columns=4, figsize=[32, 32]):
    """plot images as a panel with columns

    Args:
        im ([H, W, 3, N]): images to plot
        columns (int, optional): number of columns in the plot. Defaults to 4.
        figsize (list, optional): figure size. Defaults to [32, 32].

    Returns:
        fig : handle to figure
    """
    fig=plt.figure(figsize=figsize)    

    H, W, C, N = im.shape
    
    rows = np.ceil(N/columns)
    for i in range(1, N+1):
        fig.add_subplot(rows, columns, i)
        if(len(im.shape)==4):
            plt.imshow(np.clip(im[:,:,:,i-1], 0, 1.0))
        else:
            plt.imshow(np.clip(im, 0, 1.0))

        plt.axis('off')
    plt.show()
    
    return fig