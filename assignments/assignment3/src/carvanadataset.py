##################################################
## Deep learning crash course, assignment 3
##################################################
## Description : dataset classes for assignment 3, modified from the Kaggle Carvana challenge
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Maintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

# Kaggle Carvana challenge
# https://www.kaggle.com/c/carvana-image-masking-challenge

import sys
from pathlib import Path
import numpy as np
import matplotlib 
from tqdm import tqdm 
import time

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Project_DIR))

from util import *

class CarvanaDataset(Dataset):
    """Dataset for modified Carvana Kaggle challenge."""

    def __init__(self, image_dir, mask_dir, transform=None):
        """Initialize the dataset

        Suggested steps:
            - Search the data_dir for all *.npy file
            - Load all images
            - Load corresponding mask file from mask_dir

        Args:
            image_dir : directory to store the images
            mask_dir : directory to store the mask

        """

        self.images = []
        self.masks = []

        # *** START CODE HERE ***
        # ToDo:
        # store all images into self.images as a [3, H, W] array
        # store all masks into self.masks as a [1, H, W] array
        num_samples = 0
        image_files = os.listdir(image_dir)
        for im_file in image_files:
            if(im_file.find('.npy')!=-1):
                num_samples += 1

        t0 = time.time()
        total_num_loaded = 0
        image_files = os.listdir(image_dir)
        with tqdm(total=num_samples) as tq:
            for im_file in image_files:
                if(im_file.find('.npy')!=-1):
                    mask_file_name = os.path.join(mask_dir, im_file)
                    if(os.path.isfile(mask_file_name)):
                        # load image
                        im = np.load(os.path.join(image_dir, im_file))
                        mask = np.load(mask_file_name)

                        self.images.append(np.transpose(im, (2, 0, 1)))
                        self.masks.append(np.expand_dims(mask, axis=0))

                        total_num_loaded = len(self.images)
                    else:
                        print("Cannot find the mask file for this case ", im_file)
                t1 = time.time()
                tq.update(1)

        # *** END CODE HERE ***

        # scale the images
        self.images = [im/np.max(im) for im in self.images]

        self.transform = transform

    def __len__(self):
        """Get the number of samples in this dataset.

        Returns:
            number of samples
        """
        # *** START CODE HERE ***
        return len(self.images)
        # *** END CODE HERE ***

    def __getitem__(self, idx):
        """Get the idx sample

        Args:
            idx (int): the index of sample to get; first sample has idx being 0

        Returns:
            sample : a tuple (image, mask)
            image : [C, H, W]
            mask : [1, H, W]
        """
        # *** START CODE HERE ***
        if idx >= len(self.images):
            raise "invalid index"

        im = self.images[idx]
        mask = self.masks[idx]
        if self.transform:
            return self.transform((im, mask))

        return (im, mask)
        # *** END CODE HERE ***
        
    def __str__(self):
        str = "Carvana Dataset\n"
        str += "  Number of images: %d" % len(self.images) + "\n"
        str += "  Number of labels: %d" % len(self.masks) + "\n"
        str += "  transform : %s" % (self.transform) + "\n"
        str += "  image shape: %d %d %d" % self.images[0].shape + "\n"
        str += "  mask shape: %d %d %d" % self.masks[0].shape + "\n"

        return str

# --------------------------------------------------------

class RandomFlip1stDim(object):
    """Randomly flip the first dimension of numpy array.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (image, mask): Sample to be flipped.
        Returns:
            res: Randomly flipped sample.
        """
        if np.random.uniform() < self.p:
            return ( np.flip(img[0], axis=1).copy(), np.flip(img[1], axis=1).copy() )

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
# --------------------------------------------------------

class RandomFlip2ndDim(object):
    """Randomly flip the second dimension of numpy array.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (image, mask): Sample to be flipped.
        Returns:
            res: Randomly flipped sample.
        """
        if np.random.uniform() < self.p:
            return ( np.flip(img[0], axis=2).copy(), np.flip(img[1], axis=2).copy() )

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

# --------------------------------------------------------

def set_up_carvana_dataset(train_dir, train_mask_dir, test_dir, test_mask_dir, num_samples_validation=512, batch_size=64):
    """Set up the carvana dataset and loader

    Args:
        carvana_dataset (DataSet): carvana dataset
        num_samples_validation (int, optional): number samples in the validation sets. Defaults to 512.
        batch_size (int, optional): batch size. Defaults to 64.

    Returns:
        train_set, test_set (carvana_dataset): dataset objects for train and test
        loader_for_train, loader_for_val, loader_for_test : data loader for train, validation and test
    """
    # add some data transformation
    transform = transforms.Compose([
            RandomFlip1stDim(0.5),RandomFlip2ndDim(0.5)
    ])

    # *** START CODE HERE ***
    # set up the training set, use the transform
    train_set = CarvanaDataset(train_dir, train_mask_dir, transform=transform)
    print(train_set)
    # set up the test set
    test_set = CarvanaDataset(test_dir, test_mask_dir, transform=None)
    print(test_set)

    # split train_set for training and validation
    dataset_size = len(train_set)
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)

    train_idx, val_idx = dataset_indices[num_samples_validation:], dataset_indices[:num_samples_validation]

    # create loader for train, val, and test
    loader_for_train = DataLoader(train_set, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(train_idx))
    loader_for_val = DataLoader(train_set, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(val_idx))
    loader_for_test = DataLoader(test_set, batch_size)

    # *** END CODE HERE ***

    return train_set, test_set, loader_for_train, loader_for_val, loader_for_test

if __name__ == "__main__":
    
    # disable the interactive plotting
    matplotlib.use("agg")

    train_dir = os.path.join(Project_DIR, "../data/carvana/train_dlcc")
    train_mask_dir = os.path.join(Project_DIR, "../data/carvana/train_masks_dlcc")

    test_dir = os.path.join(Project_DIR, "../data/carvana/test_dlcc")
    test_mask_dir = os.path.join(Project_DIR, "../data/carvana/test_masks_dlcc")

    result_dir = os.path.join(Project_DIR, "../result/carvana")

    # create result folder
    os.makedirs(result_dir, exist_ok=True)

    train_set, test_set, loader_for_train, loader_for_val, loader_for_test = set_up_carvana_dataset(train_dir, train_mask_dir, test_dir, test_mask_dir, num_samples_validation=512, batch_size=16)

    # directly get one sample
    im, mask = train_set[122]
    print("Get one sample ", im.shape, mask.shape)

    # plot a batch
    columns=4
    figsize=[32, 32]

    iter_train = iter(loader_for_train)
    images, labels = iter_train.next()
    f = plot_image_array(np.transpose(images.numpy(), (2,3,1,0)), labels.numpy(), None, columns=columns, figsize=figsize)
    f.savefig(os.path.join(result_dir, "carvana_train_batch.png"), dpi=300)
    f = plot_image_array(np.transpose(labels.numpy(), (2,3,1,0)), labels.numpy(), None, columns=columns, figsize=figsize)
    f.savefig(os.path.join(result_dir, "carvana_train_batch_masks.png"), dpi=300)

    iter_val = iter(loader_for_val)
    images, labels = iter_val.next()
    f = plot_image_array(np.transpose(images.numpy(), (2,3,1,0)), labels.numpy(), None, columns=columns, figsize=figsize)
    f.savefig(os.path.join(result_dir, "carvana_val_batch.png"), dpi=300)
    f = plot_image_array(np.transpose(labels.numpy(), (2,3,1,0)), labels.numpy(), None, columns=columns, figsize=figsize)
    f.savefig(os.path.join(result_dir, "carvana_val_batch_masks.png"), dpi=300)

    iter_test = iter(loader_for_test)
    images, labels = iter_test.next()
    f = plot_image_array(np.transpose(images.numpy(), (2,3,1,0)), labels.numpy(), None, columns=columns, figsize=figsize)
    f.savefig(os.path.join(result_dir, "carvana_test_batch.png"), dpi=300)
    f = plot_image_array(np.transpose(labels.numpy(), (2,3,1,0)), labels.numpy(), None, columns=columns, figsize=figsize)
    f.savefig(os.path.join(result_dir, "carvana_test_batch_masks.png"), dpi=300)