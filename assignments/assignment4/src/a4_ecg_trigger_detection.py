##################################################
## Deep learning crash course, assignment 4
##################################################
## Description : a transformer model for ecg trigger detection
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

import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

logger = logging.getLogger(__name__)

import os
import sys
import math
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
import transformer
import ecgdataset

# get the wandb
import wandb

# disable the interactive plotting
matplotlib.use("agg")

# set seed
util.set_seed(12345)

# ----------------------------------
def add_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pytorch transformer model for ecg trigger detection")

    # parameters for training
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size') # you may need to reduce batch size to fit to your GPU
    parser.add_argument('--reg', type=float, default=0.1, help='regularization lambda')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='learning rate')
    parser.add_argument('--use_gpu', type=bool, default=True, help='if system has gpu and this option is true, will use the gpu')
    parser.add_argument('--epoch_to_load', type=int, default=-1, help='if >=0, load this check point')
    
    # parameter for data
    parser.add_argument('--sigma', type=float, default=2.0, help='sigma used to smoothing out the trigger signal')
    
    # parameter for models
    parser.add_argument('--seq_length', type=int, default=1024, help='length of sequence')
    parser.add_argument('--n_layers', type=int, default=8, help='number of transformer layers')
    parser.add_argument('--n_head', type=int, default=8, help='number of heads in attention layer')
    parser.add_argument('--n_embd', type=int, default=256, help='the embedding dimension of transformer layer')
    
    parser.add_argument(
        "--training_record",
        type=str,
        default="pytorch_ecg_triggering",
        help='String to record this training')
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help='Optimizer, sgd or adam')
    
    return parser
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
        'epoch_to_load': args.epoch_to_load,
        'use_gpu' : args.use_gpu,
        'sigma': args.sigma,
        'seq_length': args.seq_length,
        'n_layers' : args.n_layers,
        'n_head' : args.n_head,
        'n_embd' : args.n_embd
    }

result_dir = os.path.join(Project_DIR, "../result/ecg")
os.makedirs(result_dir, exist_ok=True)

training_check_point_dir = os.path.join(Project_DIR, "../result/ecg/check_points")
os.makedirs(training_check_point_dir, exist_ok=True)

data_dir = os.path.join(Project_DIR, "../data/ecg")

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

    # read in the data
    train_dir = os.path.join(Project_DIR, "../data/ecg/train")
    test_dir = os.path.join(Project_DIR, "../data/ecg/test")

    result_dir = os.path.join(Project_DIR, "../result/ecg")
    os.makedirs(result_dir, exist_ok=True)
    
    train_set, test_set, loader_for_train, loader_for_val, loader_for_test = ecgdataset.set_up_ecg_dataset(train_dir, test_dir, batch_size=config.batch_size, num_starts=10, chunk_length=config.seq_length, sigma=config.sigma, val_frac=0.1)
    
    waves, triggers, names = train_set[0]
    T, C = waves.shape
    
    # declare the model
    model = transformer.ECGDetector(n_layer=config.n_layers, 
                                    C=C, 
                                    T=T, 
                                    output_size=1, 
                                    is_causal=False, 
                                    n_embd=config.n_embd, 
                                    n_head=config.n_head, 
                                    dropout_p=0.1)
    print(model)
   
    # declare the loss function, loss_func
    loss_func = transformer.LossTriggerDetection()

    # gradient clip for temporal training
    grad_norm_clip = 1.0
    
    # a warmup stage is used
    # the total length of sequence processed is recorded
    # if the length is les than warmup_length, learning rate is scaled down for a warm start
    warmup_length=20*64*T
    # then the learning rate is decayed with cosine function
    final_length=(config.epochs-1)*len(train_set)*T
        
    # declare the optimizer, check the config.optimizer and define optimizer, note to use config.learning_rate and config.reg
    if(config.optimizer=='sgd'):
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.reg)
    else:
        #optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.reg)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.reg, amsgrad=False)
           
    # load models if needed
    if(config.epoch_to_load>=0):
        ckpt_model = os.path.join(training_check_point_dir, f"ckpt_{config.epoch_to_load}.pbt")
        print("Load check point: ", ckpt_model)
        if os.path.isfile(ckpt_model):
            state = torch.load(str(ckpt_model))
            model.load_state_dict(state['model'])
            print('Restored model, epoch {}, step {:,}'.format(state['epoch'], state['step']))

    # save model function
    save = lambda ep, step: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
    }, os.path.join(training_check_point_dir, f"ckpt_{ep}.pbt"))

    # set up training
    n_batches = len(train_set)//config.batch_size 
    
    loss_train = []
    loss_val = []
    
    best_model = None
    best_val_loss = 1e4
    
    seq_length_processed = 0
    
    # uncomment to use multiple GPUs
    # if device != torch.device('cpu') and torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    #     print(f"Train model on %d GPUs ... " % torch.cuda.device_count())
            
    model.to(device=device)
    for e in range(config.epochs):
               
        # set up the progress bar
        tq = tqdm(total=(n_batches * config.batch_size), desc ='Epoch {}, total {}'.format(e, config.epochs))

        model.train()

        t0 = time.time()
        count = 0
        running_loss_training = 0
        loss = 0
        for x, y, z in loader_for_train:

            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
          
            # *** START CODE HERE ***
            # ToDo : finish the training loop, remember to use the gradient clip
            
            # 2. perform foward pass
            output = model(x)
            
            # 3. compute loss
            loss = loss_func(output, y)
            
            # 4. zero grad
            optimizer.zero_grad()
                
            # 5. back-prop
            loss.backward()
            
            # 6. a common trick to clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
                
            # 7. update parameters
            optimizer.step()
            
            # *** END CODE HERE ***
             
            # decay the learning rate based on our progress
            lr_mult = 1
            seq_length_processed += y.shape[0]*y.shape[1]
            if seq_length_processed < warmup_length:
                # linear warmup
                lr_mult = float(seq_length_processed) / float(max(1, warmup_length))
            else:
                # cosine learning rate decay
                progress = float(seq_length_processed - warmup_length) / float(max(1, final_length - warmup_length))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = config.learning_rate * lr_mult
            
            # the learning rate can be set
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                                                   
            # bookkeeping    
            wandb.log({"batch loss":loss.item()})
            wandb.log({"batch learing rate":lr})
                
            tq.update(config.batch_size)
            tq.set_postfix({'loss':loss.item(), 'learning rate':lr, 'lr_mult':lr_mult})
            
            running_loss_training += loss.item()
            
            count += 1

        t1 = time.time()

        # save the check point
        save(e, count)
      
        # process the validation set
        model.eval()
        val_losses = []
        t0_val = time.time()
        with torch.no_grad():
            for x, y, z in loader_for_val:

                x = x.to(device=device, dtype=torch.float32)
                y = y.to(device=device, dtype=torch.float32)
            
                output = model(x)
                val_loss = loss_func(output, y)
                
                val_losses.append(val_loss.item())
            
        t1_val = time.time()
            
        loss_train.append(running_loss_training/count)
        loss_val.append(np.mean(val_losses))
        
        # keep the best model, evaluated on the validation set
        if(best_val_loss>np.mean(val_losses)):
            best_val_loss = np.mean(val_losses)
            best_model = model
        
        wandb.log({"epoch":e, "train loss":loss_train[e], "val loss":loss_val[e]})
                       
        str_after_val = '%.2f/%.2f seconds for Training/Validation - Tra loss = %.4f, Val loss = %.4f, - learning rate = %.6f' % (t1-t0, t1_val-t0_val, loss_train[e], loss_val[e], lr)
        tq.set_postfix_str(str_after_val)
        tq.close() 
        
    # apply the model on the test set
    test_bar = tqdm(enumerate(loader_for_test), total=len(loader_for_test))
    
    t0_test = time.time()
    test_x = []
    test_y = []
    test_y_hat = []
    test_names = []
    best_model.eval()
    test_loss = 0
    with torch.no_grad():
        for it, (x, y, n) in test_bar:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            
            output = best_model(x)
            loss = loss_func(output, y)
            
            test_loss += loss.item()
            
            y_hat = torch.sigmoid(output)
            
            test_x.append(x.detach().cpu().numpy())
            test_y.append(y.detach().cpu().numpy())
            test_y_hat.append(y_hat.detach().cpu().numpy())
            test_names.append(n)
            
            t1_test = time.time()    
            test_bar.set_description(f"duration: {t1_test-t0_test:.1f}, batch: {it}, loss: {loss.item():.6f}")
        
    test_loss /= len(loader_for_test)
    
    # ----------------------------------------------

    return best_model, loss_train, loss_val, (test_x, test_y, test_y_hat, test_names)

def main():
       
    moment = strftime("%Y%m%d_%H%M%S", gmtime())

    wandb.init(project="A4_Pytorch_ecg_triggering", config=config_defaults, tags=moment)
    wandb.watch_called = False

    # perform training
    m, loss_train, loss_val, test_results = run_training()

    # plot training loass
    fig=plt.figure(figsize=[8, 8])
    plt.plot(loss_train, 'b-')
    plt.plot(loss_val, 'r-')
    plt.legend(['train', 'val'])
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('training/val loss', fontsize=20)
    fig.savefig(os.path.join(result_dir, "plot_training_loss.png"), dpi=300)

    # plot a test batch
    test_x, test_y, test_y_hat, test_names = test_results
    which_batch = np.random.randint(0, len(test_x))
    triggers_found = util.adaptive_trigger_finding(test_y_hat[which_batch].squeeze(), p_thresh=0.75, d_thresh=10)
    figs = util.plot_ecg_waves(test_x[which_batch], test_y[which_batch], test_names[which_batch][0], triggers_found, figsize=[32, 32])
    
    for i, f in enumerate(figs):
        fig_name = os.path.join(result_dir, f"ecg_test_{test_names[which_batch][0][i]}_{test_names[which_batch][1][i]}.png")
        logger.info(f"save {i} -- {fig_name}")
        f.savefig(fig_name, dpi=100)
        wandb.log({"ecg test samples": f})
        plt.close(f)
        
if __name__ == '__main__':
    main()