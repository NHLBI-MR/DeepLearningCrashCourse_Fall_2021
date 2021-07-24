##################################################
## Deep learning crash course, assignment 3
##################################################
## Description : a small resnet model, using pytorch and wandb
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Mmaintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
from time import gmtime, strftime

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

from cifar10dataset import *
import util
import resnet_model
import train

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms
   
# get the wandb
import wandb

# ----------------------------------
def add_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pytorch CNN model for Cifar10 classification, using wandb")

    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--reg', type=float, default=0.005, help='regularization lambda')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='learn rate')
    parser.add_argument('--use_gpu', type=bool, default=True, help='if system has gpu and this option is true, will use the gpu')

    parser.add_argument(
        "--training_record",
        type=str,
        default="pytorch_small_resnet",
        help='String to record this training')
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help='Optimizer, sgd or adam')

    return parser
# ----------------------------------
   
num_samples_validation = 3000

# ----------------------------------

# load dataset
cifar10_dataset = util.load_and_prepare_data(os.path.join(Project_DIR, "../data/cifar10"), subtract_mean=True)

# load parameters
args = add_args().parse_args()
print(args)

config_defaults = {
        'epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'reg': args.reg,
        'use_gpu' : args.use_gpu
    }

# ----------------------------------
        
def run_training():
    """Run the training

    Outputs:
        model : best model after training
        loss_train, loss_val : loss for every epoch
        accu_train, accu_val : accuracy for every epoch
    """

    # Initialize a new wandb run
    wandb.init(config=config_defaults)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config
    print("get the config from wandb :", config)

    # prepare the datasets and loaders
    train_set, test_set, loader_for_train, loader_for_val, loader_for_test = set_up_cifar_10_dataset(cifar10_dataset, num_samples_validation=num_samples_validation, batch_size=config.batch_size)

    # get the sample size
    H, W, C, B = cifar10_dataset['X_train'].shape

    # *** START CODE HERE ***
    # declare the model m
    m = resnet_model.Cifar10SmallResNet(H, W, C)
    print(m)

    # declare the loss function, loss_func
    loss_func = nn.CrossEntropyLoss()

    # declare the optimizer, check the config.optimizer and define optimizer, note to use config.learning_rate and config.reg
    if(config.optimizer=='sgd'):
        optimizer = optim.SGD(m.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.reg)
    else:
        optimizer = optim.Adam(m.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.reg)

    # declare the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.5, last_epoch=-1, verbose=False)
    # *** START CODE HERE ***

    # get the device    
    device = util.find_GPU()
    if (config.use_gpu is False):
        device = torch.device('cpu')

    # run the training
    trainer = train.Trainer(x_dtype=torch.float32, y_dytpe=torch.long, device=device)
    m, loss_train, loss_val, accu_train, accu_val = trainer.run_training_loop(m, loss_func, optimizer, scheduler, loader_for_train, loader_for_val, wandb, config)

    # compute test accuracy
    loss_test, accu_test = trainer.compute_test_accuracy(loader_for_test, m, loss_func)

    wandb.log({"loss_test":loss_test})
    wandb.log({"accu_test":accu_test})

    return m, loss_train, loss_val, loss_test, accu_train, accu_val, accu_test

def main():

    moment = strftime("%Y%m%d_%H%M%S", gmtime())

    wandb.init(project="A3_Pytorch_small_resnet", config=config_defaults, tags=moment)
    wandb.watch_called = False

    # perform training
    m, loss_train, loss_val, loss_test, accu_train, accu_val, accu_test = run_training()

    # print out accuracy
    print('Train, validation and test accuracies are %f, %f, %f for experiment run %s' % (accu_train[args.num_epochs-1], accu_val[args.num_epochs-1], accu_test, args.training_record))
    
if __name__ == '__main__':
    main()