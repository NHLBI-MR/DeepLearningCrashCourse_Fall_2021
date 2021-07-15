##################################################
## Deep learning crash course, assignment 2
##################################################
## Description : the N-layer MLP model, using pytorch
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
from six.moves import cPickle as pickle
import argparse

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

from dataset import *
import util

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms

class PytorchMLP(nn.Module):

    def __init__(self, H, W, C, num_hidden_layers):
        """Initial the model

        Please create the pytorch layers for MLP. Please use ReLU as the nonlinear activation. 
        Hints: torch.nn.Sequential may be useful. Also, check torch.nn.Linear and torch.nn.ReLU

        Args:
            H (int): Height of input image
            W (int): Width of input image
            C (int): Number of channels of input image
            num_hidden_layers (list, optional): number of hidden layers. Defaults to [300, 300, 200, 100].
        """
        super().__init__()
        
        # *** START CODE HERE ***
        self.blocks = nn.Sequential()
        for i, num_neurons in enumerate(num_hidden_layers):
            if(i==0):
                input_dim = int(H*W*C)
            else:
                input_dim = num_hidden_layers[i-1]
                
            output_dim = num_neurons
                
            self.blocks.add_module("fc_%d" % i, nn.Linear(input_dim, output_dim, bias=True))
            self.blocks.add_module("relu_%d" % i, nn.ReLU())
                
        self.blocks.add_module("fc_output", nn.Linear(output_dim, 10, bias=True))
        
        # *** END CODE HERE ***

    def forward(self, x):
        """Forward pass of MLP model

        Args:
            x ([B, C, H, W]): a batch of input image

        Returns:
            output ([B, 10]): logits tensor, ready for the softmax
        """
        # *** START CODE HERE ***
        x = torch.flatten(x, 1)
        x = self.blocks(x)
        return x
        # *** END CODE HERE ***        

def compute_test_accuracy(loader, model, loss_func, device=torch.device('cpu')):
    
    running_loss_train = 0.0
    total = 0
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            x, y = data            
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
                                          
            y_hat = model(x)
            loss = loss_func(y_hat, y)
            
            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            running_loss_train += loss.item()

        loss = running_loss_train / (i+1)
        accu = correct / total
    
    return loss, accu
        
        
def run_training(args, cifar10_dataset, num_samples_validation=1000):
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
    num_epochs = args.num_epochs
    num_hidden_layers = args.num_hidden_layers
    batch_size = args.batch_size
    reg = args.reg
    learning_rate = args.learning_rate
    use_gpu = args.use_gpu

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

    if(args.one_batch_training):
        print("Train with only one batch")
        train_idx, val_idx = dataset_indices[num_samples_validation:num_samples_validation+batch_size], dataset_indices[:num_samples_validation]
    else:
        train_idx, val_idx = dataset_indices[num_samples_validation:], dataset_indices[:num_samples_validation]

    loader_for_train = DataLoader(train_set, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(train_idx))
    loader_for_val = DataLoader(train_set, batch_size=batch_size, sampler=sampler.SubsetRandomSampler(val_idx))

    H, W, C, B = cifar10_dataset['X_train'].shape
    
    # declare the model
    model = PytorchMLP(H, W, C, num_hidden_layers)      
    print(model)        
    
    # declare the loss function
    loss_func = nn.CrossEntropyLoss()
    
    # declare the optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=reg)
    
    # check the device
    device = util.find_GPU()
    if (use_gpu is False):
        device = torch.device('cpu')

    if device != torch.device('cpu') and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Train model on %d GPUs ... " % torch.cuda.device_count())
    
    # set model to device
    model.to(device=device)

    loss_train = []
    loss_val = []
    accu_train = []
    accu_val = []

    # train for num_epochs
    for epoch in range(num_epochs):

        model.train()
        
        # go through all mini-batches for this epoch
        running_loss_train = 0.0
        running_accu_train = 0.0
        for i, data in enumerate(loader_for_train, 0):

            x, y = data
            x = x.to(device=device, dtype=torch.float32) 
            y = y.to(device=device, dtype=torch.long) 
                          
            # *** START CODE HERE ***  
            # forward pass, put the model output to y_hat
            y_hat = model(x)

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
            
        # after one epoch, compute training loss and accuracy
        loss_train.append(running_loss_train/(i+1))
        accu_train.append(running_accu_train/(i+1))

        # after one epoch, compute validation loss and accuracy
        lv, av = compute_test_accuracy(loader_for_val, model, loss_func, device=device)
        loss_val.append(lv)
        accu_val.append(av)

        print('epoch %d, train loss %f, accuracy %f - val loss %f, accuracy %f' % (epoch, loss_train[epoch], accu_train[epoch], loss_val[epoch], accu_val[epoch]))

    # compute test accuracy
    test_set = Cifar10Dataset(cifar10_dataset['X_test'], cifar10_dataset['Y_test'], transform=None)
    loader_for_test = DataLoader(test_set, batch_size=args.batch_size)
    loss_test, accu_test = compute_test_accuracy(loader_for_test, model, loss_func, device=device)
    
    return model, loss_train, loss_val, loss_test, accu_train, accu_val, accu_test

def add_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pytorch N-layer MLP for Cifar10 classification")

    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument("--num_hidden_layers", type=int, nargs="+", default=[300, 200, 200, 200, 100])
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--reg', type=float, default=0.0025, help='regularization lambda')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learn rate')
    parser.add_argument('--use_gpu', type=bool, default=True, help='if system has gpu and this option is true, will use the gpu')

    parser.add_argument(
        "--training_record",
        type=str,
        default="pytorch_N_layer_MLP",
        help='String to record this training')

    parser.add_argument('--one_batch_training', type=bool, default=False, help='if True, train with only one batch, for debugging purpose')

    return parser

def main():

    # load parameters
    args = add_args().parse_args()
    print(args)

    # load dataset
    cifar10_dataset = util.load_and_prepare_data(os.path.join(Project_DIR, "../data"), subtract_mean=True)

    # create result folder
    os.makedirs(os.path.join(Project_DIR, "../result"), exist_ok=True)

    # perform training
    num_samples_validation = 3000
    model, loss_train, loss_val, loss_test, accu_train, accu_val, accu_test = run_training(args, cifar10_dataset, num_samples_validation)

    # print out accuracy
    print('Train, validation and test accuracies are %f, %f, %f for experiment run %s' % (accu_train[args.num_epochs-1], accu_val[args.num_epochs-1], accu_test, args.training_record))
    
    # plot the loss and accuracy curves
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(np.arange(args.num_epochs), loss_train,'r', label='train')
    ax1.plot(np.arange(args.num_epochs), loss_val, 'b', label='validation')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss')
    ax1.set_title(args.training_record)
    ax1.legend()

    ax2.plot(np.arange(args.num_epochs), accu_train,'r', label='train')
    ax2.plot(np.arange(args.num_epochs), accu_val, 'b', label='validation')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('accuracy')
    ax2.set_title(f'%s for %d hidden layers' % (args.training_record, len(args.num_hidden_layers)))
    ax2.legend()

    fig.savefig(os.path.join(Project_DIR, '../result/' + args.training_record + '.png'))
    
if __name__ == '__main__':
    main()