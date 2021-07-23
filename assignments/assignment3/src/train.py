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
import time
from pathlib import Path
from tqdm import tqdm 
from tqdm import trange
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

# ------------------------------------

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
        best_model (pytorch model) : trained best model
        loss_train, loss_val : loss during training and validation
        accu_train, accu_val : accuracy during training
    """

    # check the device
    if device != torch.device('cpu') and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Train model on %d GPUs ... " % torch.cuda.device_count())

    # set model to device
    model.to(device=device)

    loss_train = []
    loss_val = []
    accu_train = []
    accu_val = []

    best_model = model
    best_accu_val = 0.0

    # train for num_epochs
    for epoch in range(config.epochs):

        # set up the progress bar
        tq = tqdm(total=(len(loader_for_train) * config.batch_size), desc ='Epoch {}, total {}'.format(epoch, config.epochs))

        # set model to train mode
        model.train()

        # go through all mini-batches for this epoch
        running_loss_train = 0.0
        running_accu_train = 0.0
        t0 = time.time()
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

            tq.update(config.batch_size)
            tq.set_postfix(loss='{:.5f}'.format(L.item()))

        current_lr = float(scheduler.get_last_lr()[0])
        logger.log({"epoch":epoch, "current_lr":current_lr})

        t1 = time.time()

        # step the scheduler
        scheduler.step()

        # after one epoch, compute training loss and accuracy
        loss_train.append(running_loss_train/(i+1))
        accu_train.append(running_accu_train/(i+1))

        # after one epoch, compute validation loss and accuracy
        t0_val = time.time()
        lv, av = compute_test_accuracy(loader_for_val, model, loss, device=device)
        loss_val.append(lv)
        accu_val.append(av)
        t1_val = time.time()

        if(av>best_accu_val):
            best_model = model

        # log the loss_val, loss_train, accu_val, accu_train to logger
        logger.log({"epoch":epoch, "loss_val":loss_val[epoch], "loss_train":loss_train[epoch]})
        logger.log({"epoch":epoch, "accu_val":accu_val[epoch], "accu_train":accu_train[epoch]})

        str_after_val = '%.2f/%.2f seconds for Training/Validation - Tra acc = %.3f, Val acc = %.3f - Tra loss = %.4f, Val loss = %.4f, - learning rate = %.6f' % (t1-t0, t1_val-t0_val, accu_train[epoch], av, loss_train[epoch], lv, current_lr)

        tq.set_postfix_str(str_after_val)
        tq.close() 

    return best_model, loss_train, loss_val, accu_train, accu_val

def main():
    pass

if __name__ == '__main__':
    main()