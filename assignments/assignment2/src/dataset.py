##################################################
## Deep learning crash course, assignment 2
##################################################
## Description : dataset classes for assignment 2
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Mmaintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

# CIFAR 10 dataset
# https://www.cs.toronto.edu/~kriz/cifar.html

import sys
from pathlib import Path
import numpy as np
import matplotlib 

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Project_DIR))

from util import *

class Cifar10Dataset(Dataset):
    """Dataset for cifar-10."""

    def __init__(self, X, Y, transform=None):
        """Initialize the dataset

        Store the X an Y into self.images and self.labels
        Make sure self.images are in the dimension [N, C, H, W]

        Args:
            X ([32, 32, 3, N]): images
            Y ([N]): labels
        """
        # *** START CODE HERE ***
        self.images = np.transpose(X, (3, 2, 0, 1))
        self.labels = Y
        assert self.images.shape[0]==self.labels.shape[0]
        # *** END CODE HERE ***
        
        self.transform = transform
        
    def __len__(self):
        """Get the number of samples in this dataset.

        Returns:
            number of samples
        """
        # *** START CODE HERE ***
        return self.images.shape[0]
        # *** END CODE HERE ***

    def __getitem__(self, idx):
        """Get the idx sample

        Args:
            idx (int): the index of sample to get; first sample has idx being 0

        Returns:
            sample : a tuple (image, label)
        """
        # *** START CODE HERE ***
        N, C, H, W = self.images.shape
        
        if idx >= N:
            raise "invalid index"

        im = self.images[idx,:,:,:]
        if self.transform:
            # note the torchvision requires input image in [H, W, C]
            im = self.transform(np.transpose(im, (1,2,0)))

        return (im, self.labels[idx])
        # *** END CODE HERE ***
        
    def __str__(self):
        str = "Cifar 10 Dataset\n"
        str += "  Number of images: %d" % self.images.shape[0] + "\n"
        str += "  Number of labels: %d" % self.labels.shape[0] + "\n"
        str += "  transform : %s" % (self.transform) + "\n"
        str += "  image shape: %d %d %d" % self.images.shape[1:] + "\n"
            
        return str
    
if __name__ == "__main__":
    
    # disable the interactive plotting
    matplotlib.use("agg")

    # load dataset
    cifar10_dataset = load_and_prepare_data(os.path.join(Project_DIR, "../data"), subtract_mean=False)
    
    with open(os.path.join(Project_DIR, "../data", "batches.meta"), "rb") as f:
        label_names = pickle.load(f, encoding="latin1")

    # create result folder
    os.makedirs(os.path.join(Project_DIR, "../result"), exist_ok=True)

    # plot the data
    f = plot_image_array(cifar10_dataset['X_train'][:,:,:,0:16], cifar10_dataset['Y_train'][0:16], label_names['label_names'], columns=4, figsize=[32, 32])
    f.savefig(os.path.join(Project_DIR, "../result", "cifar10_images.png"), dpi=300)

    # test dataset
    train_set = Cifar10Dataset(cifar10_dataset['X_train'], cifar10_dataset['Y_train'], transform=None)
    print("Information for training set ... ", train_set)
    test_set = Cifar10Dataset(cifar10_dataset['X_test'], cifar10_dataset['Y_test'], transform=None)
    print("Information for test set ... ", test_set)

    # directly get one sample
    im, label = train_set[12]
    print("Get one sample ", im.shape)

    # create and load a batch
    batch_size = 16
    num_validation = 1000

    dataset_size = len(train_set)
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)
    train_idx, val_idx = dataset_indices[num_validation:], dataset_indices[:num_validation]

    loader_for_train = DataLoader(train_set, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(train_idx))
    loader_for_val = DataLoader(train_set, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(train_idx))
    
    # no need to shuffle the test set
    loader_for_test = DataLoader(test_set, batch_size=batch_size)

    # plot a batch
    iter_train = iter(loader_for_train)
    images, labels = iter_train.next()
    f = plot_image_array(np.transpose(images.numpy(), (2,3,1,0)), labels.numpy(), label_names['label_names'], columns=4, figsize=[32, 32])
    f.savefig(os.path.join(Project_DIR, "../result", "cifar10_train_batch.png"), dpi=300)

    iter_val = iter(loader_for_val)
    images, labels = iter_val.next()
    f = plot_image_array(np.transpose(images.numpy(), (2,3,1,0)), labels.numpy(), label_names['label_names'], columns=4, figsize=[32, 32])
    f.savefig(os.path.join(Project_DIR, "../result", "cifar10_val_batch.png"), dpi=300)

    iter_test = iter(loader_for_test)
    images, labels = iter_test.next()
    f = plot_image_array(np.transpose(images.numpy(), (2,3,1,0)), labels.numpy(), label_names['label_names'], columns=4, figsize=[32, 32])
    f.savefig(os.path.join(Project_DIR, "../result", "cifar10_test_batch.png"), dpi=300)

    # now, add some random transformation
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90.0)
    ])

    # set the transform
    train_set.transform = transform

    # now plot a batch
    images, labels = iter_train.next()
    f = plot_image_array(np.transpose(images.numpy(), (2,3,1,0)), labels.numpy(), label_names['label_names'], columns=4, figsize=[32, 32])
    f.savefig(os.path.join(Project_DIR, "../result", "cifar10_train_batch_with_random_flipping.png"), dpi=300)