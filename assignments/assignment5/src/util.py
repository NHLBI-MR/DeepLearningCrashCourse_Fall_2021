import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import random

def set_seed(seed=1234):

    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_GPU():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('using device:', device)

    return device

def dice_coeff(y_pred, y):
    a = y_pred.view(-1).float()
    b = y.view(-1).float()
    inter = torch.dot(a, b) + 0.0001
    union = torch.sum(a) + torch.sum(b) + 0.0001
    return 2*inter.float()/union.float()

def plot_image_array(im, columns=4, figsize=[8, 8]):
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
    
    rows = int(np.ceil(N/columns))
    for i in range(1, N+1):
        fig.add_subplot(rows, columns, i)
        if(len(im.shape)==4):
            plt.imshow(np.clip(im[:,:,:,i-1], 0, 1.0))
        else:
            plt.imshow(np.clip(im, 0, 1.0))

        plt.axis('off')
    
    return fig

def compute_accuracy(output, labels):
    _, pred = torch.max(output, 1)
    accuracy = (pred == labels).sum().item() * 1. / labels.shape[0]
    return accuracy

def load_cifar_10_batch(filename):
    """
    Load a batch from cifar 10 dataset

    Args:
        filename ([type]): [description]

    Returns:
        X : [32,32,3,N], N images
        Y : [N, 1], N labels
    """
    with open(filename, "rb") as f:
        datadict = pickle.load(f, encoding="latin1")
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.astype("float")
        Y = np.array(Y)
                
    return X, Y

def load_cifar_10(data_dir):
    """Load all cifar 10 images and labels

    Args:
        data_dir (string): directory to store the data

    Returns:
        X_train ([32, 32, 3, N]): training images
        Y_train ([N, 1]): training labels
        X_test ([32, 32, 3, N]): test images
        Y_test ([N, 1]): test labels
    """
    xs = []
    ys = []
    for b in range(5):
        X, Y = load_cifar_10_batch(os.path.join(data_dir, "data_batch_%d" % (b+1)))
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs)
    Y_train = np.concatenate(ys)
    
    X_test, Y_test = load_cifar_10_batch(os.path.join(data_dir, "test_batch"))
    X_train = X_train.reshape(X_train.shape[0], 3, 32, 32).transpose(2, 3, 1, 0).astype(np.float32)
    X_test = X_test.reshape(X_test.shape[0], 3, 32, 32).transpose(2, 3, 1, 0).astype(np.float32)
    
    X_train /= 255.0
    X_test /= 255.0
    
    return X_train, Y_train, X_test, Y_test

def load_and_prepare_data(data_dir, subtract_mean=True):
    """load and prepare the cifar-10 dataset

    Args:
        data_dir : the directory to store the data
        subtract_mean (bool, optional): If ture, the mean images of training set will be computed and is subtracted from training and test images. Defaults to True.

    Returns:
        dataset : a dictionary {'X_train', 'Y_train', 'X_test', 'Y_test'}
    """
    
    X_train, Y_train, X_test, Y_test = load_cifar_10(data_dir)

    mean_image = None
    if subtract_mean:
        mean_image = np.mean(X_train, axis=3, keepdims=True)
        X_train -= mean_image
        X_test -= mean_image

    return {"X_train": X_train, "Y_train": Y_train, "X_test": X_test, "Y_test": Y_test, "mean_image": mean_image}

def plot_adverserial_attack_image_array(im, label, label_names, original_label=None, columns=4, figsize=[32, 32]):
    """plot images as a panel with columns for adverserial attack testing

    Args:
        im ([H, W, 3, N]): images to plot
        columns (int, optional): number of columns in the plot. Defaults to 4.
        figsize (list, optional): figure size. Defaults to [32, 32].

        label, original_label : labels and original labels for the images
        label_names : name for every class, e.g. airplane, ship ...
    Returns:
        fig : handle to figure
    """
    fig=plt.figure(figsize=figsize)    

    H, W, C, N = im.shape
    
    rows = int(np.ceil(N/columns))
    for i in range(1, N+1):
        fig.add_subplot(rows, columns, i)
        if(len(im.shape)==4):
            plt.imshow(np.clip(im[:,:,:,i-1], 0, 1.0))
        else:
            plt.imshow(np.clip(im, 0, 1.0))
        if(label_names is not None):
            if(original_label is not None):
                plt.title(label_names[original_label[i-1]]+'->'+label_names[label[i-1]], fontsize=14)
            else:
                plt.title(label_names[label[i-1]], fontsize=20)

        plt.axis('off')
    
    return fig