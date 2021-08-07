##################################################
## Deep learning crash course, assignment 2
##################################################
## Description : the N-layer MLP model, using pytorch and wandb
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Maintainer: xueh2@github
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

from dataset import *
import util
import model

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
    parser = argparse.ArgumentParser(description="Pytorch N-layer MLP for Cifar10 classification, wandb")

    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument("--num_hidden_layers", type=int, nargs="+", default=[300, 200, 200, 200, 100])
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--reg', type=float, default=0.001, help='regularization lambda')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learn rate')
    parser.add_argument('--use_gpu', type=bool, default=True, help='if system has gpu and this option is true, will use the gpu')

    parser.add_argument(
        "--training_record",
        type=str,
        default="pytorch_N_layer_MLP",
        help='String to record this training')

    parser.add_argument(
        "--sweep_id",
        type=str,
        default="none",
        help='sweep id generated with setup script')

    return parser
# ----------------------------------
   
num_samples_validation = 3000

# ----------------------------------

# load dataset
cifar10_dataset = util.load_and_prepare_data(os.path.join(Project_DIR, "../data"), subtract_mean=True)

# create result folder
os.makedirs(os.path.join(Project_DIR, "../result"), exist_ok=True)

# load parameters
args = add_args().parse_args()
print(args)

config_defaults = {
        'seeds': 12345,
        'epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'optimizer': 'sgd',
        'num_hidden_layers': args.num_hidden_layers,
        'scheduler': 'step',
        'reg': args.reg
    }

# ----------------------------------
        
def run_training():
    """Run the training

    Inputs:
        args : arguments
        cifar10_dataset : dataset loaded with utlity functions
        num_samples_validation : number of samples for validation

    Outputs:
        model : model after training
        loss_train, loss_val : loss for every epoch
        accu_train, accu_val : accuracy for every epoch
    """

    # get the training parameters
    use_gpu = args.use_gpu

    # Initialize a new wandb run
    wandb.init(config=config_defaults)
    
    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config
    print("get the config from wandb :", config)

    # seed should be recorded as well!
    util.set_seed(int(config.seeds))

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

    loader_for_train = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler.SubsetRandomSampler(train_idx))
    loader_for_val = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler.SubsetRandomSampler(val_idx))

    H, W, C, B = cifar10_dataset['X_train'].shape
    
    # declare the model
    m = model.PytorchMLP(H, W, C, config.num_hidden_layers)      
    print(m)        
    
    # declare the loss function
    loss_func = nn.CrossEntropyLoss()
    
    # declare the optimizer, check the config.optimizer and define optimizer, note to use config.learning_rate and config.reg
    # *** START CODE HERE ***  
    if(config.optimizer=='sgd'):
        optimizer = optim.SGD(m.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.reg)
    else:
        optimizer = optim.Adam(m.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.reg)
     # *** END CODE HERE ***  
    
    # declare the scheduler, check the config.scheduler and define scheduler, check torch.optim.lr_scheduler.StepLR and torch.optim.lr_scheduler.OneCycleLR
     # *** START CODE HERE ***  
    if (config.scheduler == "step"):
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5, last_epoch=-1, verbose=False)
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.2,
                                                        total_steps=None, epochs=config.epochs,
                                                        steps_per_epoch=len(loader_for_train), pct_start=0.3,
                                                        anneal_strategy='cos', cycle_momentum=True,
                                                        base_momentum=0.85, max_momentum=0.95,
                                                        div_factor=50,
                                                        final_div_factor=10000.0,
                                                        last_epoch=-1)
     # *** END CODE HERE ***  
     
    # check the device
    device = util.find_GPU()
    if (use_gpu is False):
        device = torch.device('cpu')

    if device != torch.device('cpu') and torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)
        print(f"Train model on %d GPUs ... " % torch.cuda.device_count())
    
    # set model to device
    m.to(device=device)

    loss_train = []
    loss_val = []
    accu_train = []
    accu_val = []

    # train for num_epochs
    for epoch in range(config.epochs):

        m.train()
        
        # go through all mini-batches for this epoch
        running_loss_train = 0.0
        running_accu_train = 0.0
        for i, data in enumerate(loader_for_train, 0):

            x, y = data
            x = x.to(device=device, dtype=torch.float32) 
            y = y.to(device=device, dtype=torch.long) 
                          
            # *** START CODE HERE ***  
            # forward pass, put the model output to y_hat
            y_hat = m(x)

            # compute loss
            loss = loss_func(y_hat, y)

            # zero the parameter gradients
            optimizer.zero_grad()
        
            # backprop
            loss.backward()
            
            # perform gradient descent step
            optimizer.step()
            # *** END CODE HERE ***
            
            running_loss_train += loss.item()            
            running_accu_train += util.compute_accuracy(y_hat.detach().cpu(), y.detach().cpu())
            
            wandb.log({"batch loss":loss.item()})
            
            # one cycle scheduler
            if(config.scheduler=='one_cycle'):
                scheduler.step()                
        
        current_lr = float(scheduler.get_last_lr()[0])
        wandb.log({"epoch":epoch, "current_lr":current_lr})
        
        # step the scheduler
        if(config.scheduler=='step'):
            scheduler.step()
            
        # after one epoch, compute training loss and accuracy
        loss_train.append(running_loss_train/(i+1))
        accu_train.append(running_accu_train/(i+1))

        # after one epoch, compute validation loss and accuracy
        lv, av = model.compute_test_accuracy(loader_for_val, m, loss_func, device=device)
        loss_val.append(lv)
        accu_val.append(av)

        # log the loss_val, loss_train, accu_val, accu_train to wandb
        # *** START CODE HERE ***  
        wandb.log({"epoch":epoch, "loss_val":loss_val[epoch], "loss_train":loss_train[epoch]})
        wandb.log({"epoch":epoch, "accu_val":accu_val[epoch], "accu_train":accu_train[epoch]})
        # *** END CODE HERE ***  
        
        print('epoch %d, learning rate %f, train loss %f, accuracy %f - val loss %f, accuracy %f' % (epoch, current_lr, loss_train[epoch], accu_train[epoch], loss_val[epoch], accu_val[epoch]))
        
    # compute test accuracy
    test_set = Cifar10Dataset(cifar10_dataset['X_test'], cifar10_dataset['Y_test'], transform=None)
    loader_for_test = DataLoader(test_set, batch_size=args.batch_size)
    loss_test, accu_test = model.compute_test_accuracy(loader_for_test, m, loss_func, device=device)
    
    wandb.log({"loss_test":loss_test})
    wandb.log({"accu_test":accu_test})
        
    return m, loss_train, loss_val, loss_test, accu_train, accu_val, accu_test

def main():
    # get the sweep_id
    sweep_id = args.sweep_id

    # note the sweep_id is used to control the condition
    print("get sweep id : ", sweep_id)
    if (sweep_id != "none"):
        print("start sweep runs ...")
        wandb.agent(sweep_id, run_training, project="A2_Pytorch_MLP_Sweep", count=50)
    else:
        print("start a regular run ...")

        moment = strftime("%Y%m%d_%H%M%S", gmtime())

        wandb.init(project="A2_Pytorch_MLP_wandb", config=config_defaults, tags=moment)
        wandb.watch_called = False

        # perform training
        m, loss_train, loss_val, loss_test, accu_train, accu_val, accu_test = run_training()

        # print out accuracy
        print('Train, validation and test accuracies are %f, %f, %f for experiment run %s' % (accu_train[args.num_epochs-1], accu_val[args.num_epochs-1], accu_test, args.training_record))
    
if __name__ == '__main__':
    main()