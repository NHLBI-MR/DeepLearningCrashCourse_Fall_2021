##################################################
## Deep learning crash course, assignment 3
##################################################
## Description : a unet model for segmentation, using pytorch and wandb
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Maintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from time import gmtime, strftime

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

from carvanadataset import *
import util
import unet_model
import train

  
# get the wandb
import wandb

# disable the interactive plotting
matplotlib.use("agg")

util.set_seed()

# ----------------------------------
def add_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pytorch CNN model for Cifar10 classification, using wandb")

    parser.add_argument('--num_epochs', type=int, default=15, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--reg', type=float, default=0.0, help='regularization lambda')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--use_gpu', type=bool, default=True, help='if system has gpu and this option is true, will use the gpu')

    parser.add_argument(
        "--training_record",
        type=str,
        default="pytorch_unet",
        help='String to record this training')
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help='Optimizer, sgd or adam')

    return parser
# ----------------------------------

num_samples_validation = 512

# ----------------------------------

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

result_dir = os.path.join(Project_DIR, "../result/carvana")
os.makedirs(result_dir, exist_ok=True)

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
    train_dir = os.path.join(Project_DIR, "../data/carvana/train_dlcc")
    train_mask_dir = os.path.join(Project_DIR, "../data/carvana/train_masks_dlcc")

    test_dir = os.path.join(Project_DIR, "../data/carvana/test_dlcc")
    test_mask_dir = os.path.join(Project_DIR, "../data/carvana/test_masks_dlcc")

    train_set, test_set, loader_for_train, loader_for_val, loader_for_test = set_up_carvana_dataset(train_dir, train_mask_dir, test_dir, test_mask_dir, num_samples_validation=num_samples_validation, batch_size=config.batch_size)

    # get the sample size
    C, H, W = train_set.images[0].shape
    print('sample image has the shape [%d, %d, %d]' % (H, W, C))

    # *** START CODE HERE ***
    # declare the model m
    m = unet_model.UNet(H, W, C)
    print(m)

    # declare the loss function, loss_func
    loss_func = unet_model.LossBinarySegmentation()

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
    x_dtype=torch.float32
    y_dtype=torch.float32
    trainer = train.UnetTrainer(x_dtype=x_dtype, y_dtype=y_dtype, device=device)
    m, loss_train, loss_val, accu_train, accu_val = trainer.run_training_loop(m, loss_func, optimizer, scheduler, loader_for_train, loader_for_val, wandb, config)

    # compute test accuracy
    loss_test, accu_test = trainer.compute_test_accuracy(loader_for_test, m, loss_func)

    wandb.log({"loss_test":loss_test})
    wandb.log({"accu_test":accu_test})

    # run on a test batch
    iter_test = iter(loader_for_test)
    test_images, test_masks = iter_test.next()

    m.eval()
    with torch.no_grad():
        x = test_images.to(device=device, dtype=x_dtype)
        y = test_masks.to(device=device, dtype=y_dtype)

        y_hat = m(x)
        correct = trainer.compute_accuracy(y_hat.detach().cpu(), y.detach().cpu())

    print("For the selected test batch, the mean dice ratio is ", correct)

    probs = torch.sigmoid(y_hat.detach().cpu())

    # plot test batch
    columns = 4
    figsize=[32, 32]

    f = plot_image_array(np.transpose(test_images[0:16, :].cpu().numpy(), (2,3,1,0)), test_masks.numpy(), None, columns=columns, figsize=figsize)
    f.savefig(os.path.join(result_dir, "carvana_test_batch_for_trained_model.png"), dpi=300)
    wandb.log({"carvana_test_batch_for_trained_model":f})

    f = plot_image_array(np.transpose(test_masks[0:16, :].cpu().numpy(), (2,3,1,0)), test_masks.numpy(), None, columns=columns, figsize=figsize)
    f.savefig(os.path.join(result_dir, "carvana_test_batch_masks_for_trained_model.png"), dpi=300)
    wandb.log({"carvana_test_batch_masks_for_trained_model":f})

    f = plot_image_array(np.transpose(probs[0:16, :].cpu().numpy(), (2,3,1,0)), test_masks.numpy(), None, columns=columns, figsize=figsize)
    f.savefig(os.path.join(result_dir, "carvana_test_batch_results_for_trained_model.png"), dpi=300)
    wandb.log({"carvana_test_batch_results_for_trained_model":f})

    return m, loss_train, loss_val, loss_test, accu_train, accu_val, accu_test

def main():

    moment = strftime("%Y%m%d_%H%M%S", gmtime())

    wandb.init(project="A3_Pytorch_unet", config=config_defaults, tags=moment)
    wandb.watch_called = False

    # perform training
    m, loss_train, loss_val, loss_test, accu_train, accu_val, accu_test = run_training()

    # print out accuracy
    print('Train, validation and test accuracies are %f, %f, %f for experiment run %s' % (accu_train[args.num_epochs-1], accu_val[args.num_epochs-1], accu_test, args.training_record))
    
    # save the best model
    model_file = os.path.join(result_dir, 'A3_Pytorch_unet_model_'+moment+'.pt')
    torch.save({'model': m.state_dict(), 'training_time':moment, 'args':args}, model_file)

    # save the model
    model_file = os.path.join(result_dir, 'A3_Pytorch_unet_model_'+moment+'.pt')
    print("Save model to ", model_file)
    torch.save({'model': m.state_dict(), 'training_time':moment, 'args':args}, model_file)

if __name__ == '__main__':
    main()