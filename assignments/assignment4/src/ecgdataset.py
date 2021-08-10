##################################################
## Deep learning crash course, assignment 4
##################################################
## Description : dataset classes for assignment 4
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Maintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

import sys
from pathlib import Path
import numpy as np
import matplotlib 
from tqdm import tqdm 
import time

import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms
from scipy.ndimage import gaussian_filter1d

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.append(str(Project_DIR))

from util import *

class ECGDataset(Dataset):
    """Dataset for ECG trigger detection."""

    def __init__(self, data_dir, data_files, chunk_length=1024, num_starts=5, transform=None):
        """Initialize the dataset

        Args:
            data_dir : directory to store the ECG waveform
            data_files : data files, a list
            chunk_length : the number of time points in each sample
            num_starts : for every ecg signal, the starting point for sampling is randomly picked for num_starts times

        """

        self.chunk_length = chunk_length
        self.num_starts = num_starts
        self.ecg_waves = []
        self.ecg_triggers = []
        self.ecg_names = []

        t0 = time.time()
        num_samples = len(data_files)*num_starts

        with tqdm(total=num_samples) as tq:
            for data_file in data_files:
                data_file_name = os.path.join(data_dir, data_file)
                ecg = np.load(data_file_name)
                self.sample_one_signal(ecg, data_file)
                tq.update(num_starts)

        t1 = time.time()
        tq.set_postfix_str(f"Data loading - {t1-t0} seconds")
        
        self.transform = transform

    def sample_one_signal(self, ecg, data_file):
        """Sample one signal

        Args:   
            ecg ([T, 7]): ecg wave form, ecg[:, 0:5] are four channels of ecg waveform; ecg[:,4] trigger signal (>=0 trigger); ecg[:,5], time; ecg[:,6], PCA signal of four channels
            
            All ecg signal was scaled by computing the mean and SD per channel            
        """
        
        T, _ = ecg.shape
        
        if(T<self.chunk_length):
            logger.info(f"{data_file}, T<self.chunk_length, {T}, {self.chunk_length}")
            return
        
        ecg_waves = ecg[:, 0:4]
        ecg_trigger = ecg[:, 4]
        
        mean_ecg = np.mean(ecg_waves, axis=0, keepdims=True)
        std_ecg = np.std(ecg_waves, axis=0, keepdims=True)
        
        ecg_normalized = (ecg_waves - mean_ecg) / std_ecg
        
        locs = np.argwhere(ecg_trigger>0)
        ecg_trigger[locs] = 1.0
                
        starting_locs = np.random.randint(locs[0], locs[-1], size=self.num_starts)
        starting_locs[0] = 0 # always start from beginning
        
        waves = []
        triggers = []
        for n in range(self.num_starts):
            s = starting_locs[n]
            inds = np.arange(s, T, self.chunk_length)
            for ind in inds:
                if(ind+self.chunk_length>=T):
                    ind = T - self.chunk_length - 1
                    
                if(ind<0):
                    ind = 0
                    
                w = ecg_normalized[ind:ind+self.chunk_length, :]
                t = ecg_trigger[ind:ind+self.chunk_length]
            
                self.ecg_waves.append(w)
                self.ecg_triggers.append(t)
                self.ecg_names.extend([(data_file, ind)])

    def __len__(self):
        """Get the number of samples in this dataset.

        Returns:
            number of samples
        """
        # *** START CODE HERE ***
        return len(self.ecg_waves)
        # *** END CODE HERE ***

    def __getitem__(self, idx):
        """Get the idx sample

        Args:
            idx (int): the index of sample to get; first sample has idx being 0

        Returns:
            sample : a tuple (ecg_signal, ecg_trigger)
            ecg_signal : [chunk_length, C]
            ecg_trigger : [chunk_length]
        """
        # *** START CODE HERE ***
        if idx >= len(self.ecg_waves):
            raise "invalid index"

        ecg_signal = self.ecg_waves[idx]
        ecg_trigger = self.ecg_triggers[idx]
        ecg_name = self.ecg_names[idx]
        
        if self.transform:
            return self.transform((ecg_signal, ecg_trigger, ecg_name))

        return (ecg_signal, ecg_trigger, ecg_name)
        # *** END CODE HERE ***
        
    def __str__(self):
        str = "ECG Dataset\n"
        str += "  Number of signals: %d" % len(self.ecg_waves) + "\n"
        str += "  Number of triggers: %d" % len(self.ecg_triggers) + "\n"
        str += "  transform : %s" % (self.transform) + "\n"
        str += "  ecg_waves shape: %d %d" % self.ecg_waves[0].shape + "\n"
        str += "  ecg_triggers shape: %d"  % self.ecg_triggers[0].shape + "\n"

        return str

# --------------------------------------------------------

class SmoothingTrigger(object):
    """Smooth the trigger signal with a gaussian filter
    Args:
        sigma (float): sigma to be used
    """

    def __init__(self, sigma=2.0):
        self.sigma = sigma

    def __call__(self, sample):
        """
        Args:
            sample (ecg_wave, trigger): Sample.
        Returns:
            res: the tuple with trigger smoothed
        """

        trigger = sample[1]
        if(np.max(trigger)>0):
            trigger_smoothed = gaussian_filter1d(trigger, sigma=self.sigma)
            trigger_smoothed /= np.max(trigger_smoothed)

            return (sample[0], trigger_smoothed, sample[2])
        else:
            return sample

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)
    
# --------------------------------------------------------

def set_up_ecg_dataset(train_dir, test_dir, batch_size=64, num_starts=20, chunk_length=2048, sigma=2.0, val_frac=0.1):
    """Set up the ecg dataset and loader

    Args:
        train_dir (str): data directory for training
        test_dir (str): data directory for testing
        batch_size (int): batch size
        num_starts (int, optional): number of staring locations to sample ecg waveform
        chunk_length (int, optional): chunk size for every sample
        sigma (float, optional): if > 0, smoothing the triggering signal with guassian filter
        val_frac (float, optional): fraction of validation signal
    Returns:
        train_set, test_set (ecg_dataset): dataset objects for train and test
        loader_for_train, loader_for_val, loader_for_test : data loader for train, validation and test
    """
    # add some data transformation
    transform = transforms.Compose([
            SmoothingTrigger(sigma)
    ])

    # get all data files
    data_files = os.listdir(train_dir)
    test_data_files = os.listdir(test_dir)

    random.shuffle(data_files)

    num_train = int((1.0-val_frac) * len(data_files))
    train_data_files = data_files[0:num_train]
    val_data_files = data_files[num_train:]

    # set up the training set, use the transform
    train_set = ECGDataset(train_dir, train_data_files, chunk_length=chunk_length, num_starts=num_starts, transform=transform)
    print(train_set)
    val_set = ECGDataset(train_dir, val_data_files, chunk_length=chunk_length, num_starts=5, transform=transform)
    print(val_set)
    
    test_set = ECGDataset(test_dir, test_data_files, chunk_length=chunk_length, num_starts=5, transform=None)
    print(test_set)
    
    # set up the test setplt.plot(ind, waves[b, ind, c], 'ro');
    # create loader for train, val, and test
    loader_for_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    loader_for_val = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    loader_for_test = DataLoader(test_set, batch_size, pin_memory=True)

    return train_set, test_set, loader_for_train, loader_for_val, loader_for_test

if __name__ == "__main__":
    
    # disable the interactive plotting
    matplotlib.use("agg")

    train_dir = os.path.join(Project_DIR, "../data/ecg/train")
    test_dir = os.path.join(Project_DIR, "../data/ecg/test")

    result_dir = os.path.join(Project_DIR, "../result/ecg/dataset")
    os.makedirs(result_dir, exist_ok=True)

    train_set, test_set, loader_for_train, loader_for_val, loader_for_test = set_up_ecg_dataset(train_dir, test_dir, batch_size=16, num_starts=20, chunk_length=512, sigma=2.0, val_frac=0.1)

    # directly get one sample
    waves, triggers, names = train_set[356]
    print("Get one sample ", waves.shape, triggers.shape, names)

    # plot a batch
    columns=4
    figsize=[32, 32]

    iter_train = iter(loader_for_train)
    waves, triggers, names = iter_train.next()    
    figs = plot_ecg_waves(waves, triggers, names[0], figsize=[16, 12])
    for i, f in enumerate(figs):
        fig_name = os.path.join(result_dir, f"ecg_train_{names[0][i]}_{names[1][i]}.png")
        logger.info(f"save {i} -- {fig_name}")
        f.savefig(fig_name, dpi=100)
        plt.close(f)