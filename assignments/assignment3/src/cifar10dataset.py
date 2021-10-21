##################################################
## Deep learning crash course, assignment 3
##################################################
## Description : dataset classes for assignment 3
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Maintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

# CIFAR 10 dataset
# https://www.cs.toronto.edu/~kriz/cifar.html
# It is the same as the assignment 2

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms

import sys
from pathlib import Path
import numpy as np
import matplotlib 

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
    
def set_up_cifar_10_dataset(cifar10_dataset, num_samples_validation=3000, batch_size=64, with_sampler=True):
    """Set up the cifar 10 dataset

    Args:
        cifar10_dataset (DataSet): cifar 10 dataset
        num_samples_validation (int, optional): number samples in the validation sets. Defaults to 3000.
        batch_size (int, optional): batch size. Defaults to 64.

    Returns:
        train_set, test_set (Cifar10Dataset): dataset objects for train and test
        loader_for_train, loader_for_val, loader_for_test : data loader for train, validation and test
    """
    # add some data transformation
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5)
    ])
           
    # set up the data loader
    train_set = Cifar10Dataset(cifar10_dataset['X_train'], cifar10_dataset['Y_train'], transform=transform)
    # do not add data augmentation to test set !
    test_set = Cifar10Dataset(cifar10_dataset['X_test'], cifar10_dataset['Y_test'], transform=None)
    
    # create and load a batch    
    dataset_size = len(train_set)
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)

    train_idx, val_idx = dataset_indices[num_samples_validation:], dataset_indices[:num_samples_validation]

    loader_for_train = DataLoader(train_set, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(train_idx), num_workers=4)
    loader_for_val = DataLoader(train_set, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(val_idx), num_workers=4)
    loader_for_test = DataLoader(test_set, batch_size)

    return train_set, test_set, loader_for_train, loader_for_val, loader_for_test

if __name__ == "__main__":
    
    # disable the interactive plotting
    matplotlib.use("agg")

    data_dir = os.path.join(Project_DIR, "../data/cifar10")
    result_dir = os.path.join(Project_DIR, "../result/cifar10")

    # load dataset
    cifar10_dataset = load_and_prepare_data(data_dir, subtract_mean=False)
    
    with open(os.path.join(data_dir, "batches.meta"), "rb") as f:
        label_names = pickle.load(f, encoding="latin1")

    # create result folder
    os.makedirs(result_dir, exist_ok=True)

    # plot the data
    f = plot_image_array(cifar10_dataset['X_train'][:,:,:,0:16], cifar10_dataset['Y_train'][0:16], label_names['label_names'], columns=4, figsize=[32, 32])
    f.savefig(os.path.join(result_dir, "cifar10_images.png"), dpi=300)

    train_set, test_set, loader_for_train, loader_for_val, loader_for_test = set_up_cifar_10_dataset(cifar10_dataset, num_samples_validation=3000, batch_size=64)

    # directly get one sample
    im, label = train_set[12]
    print("Get one sample ", im.shape)

    # plot a batch
    iter_train = iter(loader_for_train)
    images, labels = iter_train.next()
    f = plot_image_array(np.transpose(images.numpy(), (2,3,1,0)), labels.numpy(), label_names['label_names'], columns=8, figsize=[32, 32])
    f.savefig(os.path.join(result_dir, "cifar10_train_batch.png"), dpi=300)

    iter_val = iter(loader_for_val)
    images, labels = iter_val.next()
    f = plot_image_array(np.transpose(images.numpy(), (2,3,1,0)), labels.numpy(), label_names['label_names'], columns=8, figsize=[32, 32])
    f.savefig(os.path.join(result_dir, "cifar10_val_batch.png"), dpi=300)

    iter_test = iter(loader_for_test)
    images, labels = iter_test.next()
    f = plot_image_array(np.transpose(images.numpy(), (2,3,1,0)), labels.numpy(), label_names['label_names'], columns=8, figsize=[32, 32])
    f.savefig(os.path.join(result_dir, "cifar10_test_batch.png"), dpi=300)