##################################################
## Deep learning crash course, assignment 3
##################################################
## Description : training loops
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Mmaintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

import util
   
# ----------------------------------
        
def run_training_loop(model, loss, optimizer, scheduler, loader_for_train, loader_for_val, logger, config, device):
    """run the training loop

    Args:
        model (pytorch model): model to train
        loss (pytorch loss) : loss to be minimized
        optimizer (pytorch optimier): optimizer used in training loop
        scheduler (scheduler for learning rate): if not none, use the scheduler to adjust learning rate
        loader_for_train, loader_for_val : data loader for train and validation
        logger (wandb): log the training process.
        config : configuration for training parameters
        device : device to train the model

    Returns:
        model (pytorch model) : trained model
        loss_train, loss_val : loss during training and validation
        accu_train, accu_val : accuracy during training
    """
     
    # check the device
    if device != torch.device('cpu') and torch.cuda.device_count() > 1:
        m = nn.DataParallel(m)
        print(f"Train model on %d GPUs ... " % torch.cuda.device_count())
    
    # set model to device
    model.to(device=device)

    loss_train = []
    loss_val = []
    accu_train = []
    accu_val = []

    # train for num_epochs
    for epoch in range(config.epochs):

        model.train()
        
        # go through all mini-batches for this epoch
        running_loss_train = 0.0
        running_accu_train = 0.0
        for i, data in enumerate(loader_for_train, 0):

            x, y = data
            x = x.to(device=device, dtype=torch.float32) 
            y = y.to(device=device, dtype=torch.long) 
                          
            # forward pass, put the model output to y_hat
            y_hat = model(x)

            # compute loss
            L = loss(y_hat, y)

            # zero the parameter gradients
            optimizer.zero_grad()
        
            # backprop
            L.backward()
            
            # perform gradient descent step
            optimizer.step()
            
            running_loss_train += L.item()            
            running_accu_train += util.compute_accuracy(y_hat.detach().cpu(), y.detach().cpu())
            
            logger.log({"batch loss":L.item()})                         
        
        current_lr = float(scheduler.get_last_lr()[0])
        logger.log({"epoch":epoch, "current_lr":current_lr})
        
        # step the scheduler
        scheduler.step()
            
        # after one epoch, compute training loss and accuracy
        loss_train.append(running_loss_train/(i+1))
        accu_train.append(running_accu_train/(i+1))

        # after one epoch, compute validation loss and accuracy
        lv, av = model.compute_test_accuracy(loader_for_val, m, loss, device=device)
        loss_val.append(lv)
        accu_val.append(av)

        # log the loss_val, loss_train, accu_val, accu_train to logger
        logger.log({"epoch":epoch, "loss_val":loss_val[epoch], "loss_train":loss_train[epoch]})
        logger.log({"epoch":epoch, "accu_val":accu_val[epoch], "accu_train":accu_train[epoch]})        
        print('epoch %d, learning rate %f, train loss %f, accuracy %f - val loss %f, accuracy %f' % (epoch, current_lr, loss_train[epoch], accu_train[epoch], loss_val[epoch], accu_val[epoch]))
               
    return model, loss_train, loss_val, accu_train, accu_val

def main():
    pass
    
if __name__ == '__main__':
    main()