##################################################
## Deep learning crash course, assignment 4
##################################################
## Description : a RNN model for character-level language model
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Maintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
## Ref to : http://karpathy.github.io/2015/05/21/rnn-effectiveness/
## Ref to : https://www.kaggle.com/francescapaulin/character-level-lstm-in-pytorch
##################################################

import torch
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
import time
from time import gmtime, strftime
from tqdm import tqdm 

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

import util
import model
 
# get the wandb
import wandb

# disable the interactive plotting
matplotlib.use("agg")

# set seed
util.set_seed(12345)

# ----------------------------------
def add_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pytorch RNN model for character level language model")

    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--reg', type=float, default=0.0, help='regularization lambda')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--use_gpu', type=bool, default=True, help='if system has gpu and this option is true, will use the gpu')
    parser.add_argument('--seq_length', type=int, default=256, help='length of sequence')
    parser.add_argument('--n_hidden', type=int, default=512, help='internal state dimension')
    parser.add_argument('--n_layers', type=int, default=4, help='number of layers for RNN')
    parser.add_argument('--n_batch_for_para_update', type=int, default=2, help='the number of batch to do one parameter update')
    
    parser.add_argument(
        "--training_record",
        type=str,
        default="pytorch_char_rnn",
        help='String to record this training')
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help='Optimizer, sgd or adam')
    
    parser.add_argument(
        "--rnn",
        type=str,
        default="lstm",
        help='lstm or gru')

    return parser
# ----------------------------------

# load parameters
args = add_args().parse_args()
print(args)

config_defaults = {
        'epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'seq_length': args.seq_length,
        'learning_rate': args.learning_rate,
        'optimizer': args.optimizer,
        'reg': args.reg,
        'use_gpu' : args.use_gpu,
        'n_hidden' : args.n_hidden,
        'n_layers' : args.n_layers,
        'rnn' : args.rnn,
        'n_batch_for_para_update' : args.n_batch_for_para_update
    }

result_dir = os.path.join(Project_DIR, "../result/char_rnn")
os.makedirs(result_dir, exist_ok=True)

data_dir = os.path.join(Project_DIR, "../data/charRNN")

device = util.find_GPU()

# ----------------------------------
                          
def run_training():
    """Run the training

    Outputs:
        model : best model after training
        loss_train : loss for every epoch
    """

    # set the seed
    util.set_seed(124)

    # Initialize a new wandb run
    wandb.init(config=config_defaults)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config
    print("get the config from wandb :", config)

    # *** START CODE HERE ***
    # read '3253.txt',  from data/charRNN folder as a long string in text
    # this book is "The Papers and Writings of Abraham Lincoln, Complete by Abraham Lincoln"
    # there are a few other books in the data_dir, you can use them as well.
    text = ''
    #txt_files = ['12233-0.txt',  '3253.txt',  '4367.txt',  '5851.txt']
    txt_files = ['3253.txt']
    for fname in txt_files:
        txt_file = os.path.join(data_dir, fname)
        with open(txt_file, 'r') as f:
            a_text = f.read()
        
        text += a_text
    # *** END CODE HERE ***
    
    # get the character encoding
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    # encode the text
    data = np.array([char2int[ch] for ch in text])

    # declare the model
    if(config.rnn == 'lstm'):
        m = model.char_rnn_lstm(chars, config.n_hidden, config.n_layers)
    else:
        m = model.char_rnn_gru(chars, config.n_hidden, config.n_layers)
        
    print(m)
   
    # declare the loss function, loss_func
    loss_func = nn.CrossEntropyLoss()

    # declare the optimizer, check the config.optimizer and define optimizer, note to use config.learning_rate and config.reg
    if(config.optimizer=='sgd'):
        optimizer = optim.SGD(m.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.reg)
    else:
        optimizer = optim.Adam(m.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.reg)

    # declare the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.8, last_epoch=-1, verbose=False)
    
    # create training and validation data
    val_idx = int(len(data)*0.9)
    data, val_data = data[:val_idx], data[val_idx:]
    
    n_batches = len(data)//(config.batch_size * config.seq_length) 
    
    loss_train = []
    loss_val = []
    
    m.to(device=device)
    for e in range(config.epochs):
        
        # initialize hidden and cell state
        h = m.init_hidden(config.batch_size, device=device)
        
        # set up the progress bar
        tq = tqdm(total=(n_batches * config.batch_size), desc ='Epoch {}, total {}'.format(e, config.epochs))
            
        m.train()
                        
        t0 = time.time()
        count = 0
        running_loss_training = 0
        loss = 0
        for x, y in util.get_batches(data, config.batch_size, config.seq_length):
                       
            # 1. one-hot-encoding x
            x = util.one_hot_encode(x, m.n_tokens)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)           
            inputs, targets = inputs.to(device=device), targets.to(device=device)
          
            # 2. perform foward pass
            output, h = m(inputs, h)
            
            # 3. compute loss
            loss += loss_func(output, torch.flatten(targets))

            # Note here: since RNN uses the hidden state to pass information across time points, we need to decide
            # the length to look back in "history"
            # As a practical limitation, we cannot do backprog through the entire history for a very long sequence,
            # so here we use the n_batch_for_para_update to decide how many batches we will look back
            # loss is accumulated until n_batch_for_para_update batches are processed and backprop is performed once
            # this is temporal truncation, a common trick in training sequence model.
            if(count>0 and (count % config.n_batch_for_para_update == 0)):
            
                # 4. zero grad
                optimizer.zero_grad()
                    
                # 5. back-prop
                loss.backward()
            
                # 6. a common trick to clip gradient
                nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                
                # 7. update parameters
                optimizer.step()
                                            
                wandb.log({"batch loss":loss.item()/config.n_batch_for_para_update})
                
                tq.update(config.batch_size*config.n_batch_for_para_update)
                tq.set_postfix(loss='{:.5f}'.format(loss.item()/config.n_batch_for_para_update))
                
                running_loss_training += loss.item()
                
                # now we reset status, so backprop will not go back further in history
                loss = 0
                h = m.init_hidden(config.batch_size, device=device)

            count += 1

        t1 = time.time()

        current_lr = float(scheduler.get_last_lr()[0])

        # step the scheduler
        scheduler.step()

        # Get validation loss
        val_h = m.init_hidden(config.batch_size, device=device)
        val_losses = []
        m.eval()

        t0_val = time.time()
        with torch.no_grad():
            for x, y in util.get_batches(val_data, config.batch_size, config.seq_length):

                x = util.one_hot_encode(x, m.n_tokens)
                x, y = torch.from_numpy(x), torch.from_numpy(y)
               
                if(type(h) is tuple):
                    val_h = tuple([each.data for each in val_h])
                else:
                    val_h = val_h.data
                
                inputs, targets = x, y
                inputs, targets = inputs.to(device=device), targets.to(device=device)

                output, val_h = m(inputs, val_h)
                val_loss = loss_func(output, torch.flatten(targets))
                
                val_losses.append(val_loss.item())
            
        t1_val = time.time()
            
        loss_train.append(running_loss_training/count)
        loss_val.append(np.mean(val_losses))
        
        wandb.log({"epoch":e, "train loss":loss_train[e], "val loss":loss_val[e]})
        
        content = util.sample(m, 512, prime='We, the people ', top_k=5, device=device)
                
        str_after_val = '%.2f/%.2f seconds for Training/Validation - Tra loss = %.4f, Val loss = %.4f, - learning rate = %.6f' % (t1-t0, t1_val-t0_val, loss_train[e], loss_val[e], current_lr)
        tq.set_postfix_str(str_after_val)
        tq.close() 
        
        print(" --> sampled texts for this epoch :   ", content)
        
    # ----------------------------------------------

    return m, loss_train, loss_val

def main():

    moment = strftime("%Y%m%d_%H%M%S", gmtime())

    wandb.init(project="A4_Pytorch_char_rnn", config=config_defaults, tags=moment)
    wandb.watch_called = False

    # perform training
    m, loss_train, loss_val = run_training()

    # plot training loass
    fig=plt.figure(figsize=[8, 8])
    plt.plot(loss_train, 'b-')
    plt.plot(loss_val, 'r-')
    plt.legend(['train', 'val'])
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('training/val loss', fontsize=20)
    fig.savefig(os.path.join(result_dir, "plot_training_loss.png"), dpi=300)

    # sample the model
    for i in range(3):
        content = util.sample(m, 4086, prime='We, the people ', top_k=5, device=device)
        print(content)
        
        f = open(os.path.join(result_dir, f"char_rnn_samples_{i}.txt"),"w+")
        f.write(content)
        f.close() 
        
if __name__ == '__main__':
    main()