##################################################
## Deep learning crash course, assignment 5
##################################################
## Description : a small GAN network, using pytorch and wandb
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Maintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm 
from time import gmtime, strftime

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

import util
import gan
  
# get the wandb
import wandb

# disable the interactive plotting
matplotlib.use("agg")

# ----------------------------------
def add_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="A small GAN netwrok, using wandb")

    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--reg', type=float, default=0.0, help='regularization lambda')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--use_gpu', type=bool, default=True, help='if system has gpu and this option is true, will use the gpu')
    parser.add_argument('--optim_step_D', type=int, default=2, help='number of optimization steps of discriminator in one iteration')
    parser.add_argument('--wgan_lambda', type=float, default=10, help='regularization for gradient penalty in WGAN-GP')

    parser.add_argument(
        "--training_record",
        type=str,
        default="pytorch_small_gan",
        help='String to record this training')
    
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        help='Optimizer, sgd or adam')
    
    parser.add_argument(
        "--loss_type",
        type=str,
        default="gan",
        help='GAN loss, gan for vanilla JSD loss; wgan for Wasserstein GAN loss with gradient penalty')

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
        'use_gpu': args.use_gpu,
        'optim_step_D': args.optim_step_D,
        'loss_type': args.loss_type,
        'wgan_lambda': args.wgan_lambda
    }

data_dir = os.path.join(Project_DIR, "../data/fashion_mnist")

if(args.loss_type=='gan'):
    result_dir = os.path.join(Project_DIR, "../result/small_gan")
else:
    result_dir = os.path.join(Project_DIR, "../result/small_wgan")
    
os.makedirs(result_dir, exist_ok=True)

# latent vector dimension
Dim = 64

# ----------------------------------
        
def run_training():
    """Run the training

    Outputs:
        model_G, model_D : G and D after training
        G_loss, D_loss : recorded training loss for G and D
    """

    # Initialize a new wandb run
    wandb.init(config=config_defaults)

    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config
    print("get the config from wandb :", config)

    # prepare the datasets and loaders
    # Note we are using the torchvision datasets
    transforms = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((.5,), (.5,))]
                )
    
    dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=transforms)
    loader_for_train = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    # get the sample size
    C, H, W = dataset[0][0].shape
    print('sample image has the shape [%d, %d, %d]' % (H, W, C))

    # get the device    
    device = util.find_GPU()
    if (config.use_gpu is False):
        device = torch.device('cpu')

    # *** START CODE HERE ***
    # declare the generator and discriminator
    G = gan.Generator(D=Dim).to(device=device)
    D = gan.Discriminator(H, W, C).to(device=device)

    # declare the optimizer, check the config.optimizer and define optimizer, note to use config.learning_rate and config.reg
    if(config.optimizer=='sgd'):
        optimizer_G = optim.SGD(G.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.reg)
        optimizer_D = optim.SGD(D.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.reg)
    else:
        optimizer_G = optim.Adam(G.parameters(), lr=config.learning_rate, betas=(0.5, 0.999), eps=1e-08, weight_decay=config.reg)
        optimizer_D = optim.Adam(D.parameters(), lr=config.learning_rate, betas=(0.5, 0.999), eps=1e-08, weight_decay=config.reg)

    # declare the scheduler
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, 5, gamma=0.5, last_epoch=-1, verbose=False)
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, 5, gamma=0.5, last_epoch=-1, verbose=False)

    # run the training
    x_dtype=torch.float32
    y_dtype=torch.float32

    # run the GAN training
    
    # uncomment this to help debug
    # torch.autograd.set_detect_anomaly(True)
    
    G_loss = []
    D_loss = []
    
    # sample the generator 2 times per epoch
    step_to_sampling = int(len(loader_for_train)/2)
    if(step_to_sampling<50):
        step_to_sampling = 50
        
    num_steps = 0
    for epoch in range(config.epochs):   
        
        tq = tqdm(total=(len(loader_for_train) * config.batch_size), desc ='Epoch {}, total {}'.format(epoch, config.epochs))
        
        G.train()
        D.train()
             
        for i, data in enumerate(loader_for_train, 0):
    
            x, y = data
            x = x.to(device=device, dtype=x_dtype) 
            y = y.to(device=device, dtype=y_dtype) 
                
            for s in range(config.optim_step_D):
                # create latent vectors
                z = torch.randn(config.batch_size, Dim, device=device)
            
                # compute ligits for real and fake samples
                logits_real = D(x)
                x_fake = G(z)
                logits_fake = D(x_fake)

                # compute loss for D
                if(args.loss_type=='gan'):
                    l_d = gan.loss_D(logits_real, logits_fake)
                else:
                    # this part is a bit involved, so it is provided to you
                    # but please read and understand what is going on here
                    
                    # compute alpha
                    alpha = torch.rand((config.batch_size, 1, 1, 1), device=device)
                    
                    # compute linear combination of real and fake samples
                    s = (alpha * x_fake + (1-alpha) * x).requires_grad_(True)
                    logits_s = D(s)
                    
                    # compute gradient
                    grad_s = torch.autograd.grad(outputs=logits_s, inputs=s, grad_outputs=torch.ones_like(logits_s), create_graph=True, only_inputs=True)[0]
                    grad_s = grad_s.view(grad_s.size(0), -1)
                    # take the L2 norm and compute the GP penalty
                    grad_s_norm = torch.norm(grad_s, dim=1, keepdim=True)
                    reg_s = (grad_s_norm-1) ** 2
                    
                    # now we can get the WGAN-GP loss
                    l_d = gan.wgan_loss_D(logits_real, logits_fake, reg_s, args.wgan_lambda)

                # optimize D
                optimizer_D.zero_grad()
                l_d.backward(retain_graph=True)
                optimizer_D.step()

                l_d_value = l_d.detach().cpu().item()
                D_loss.append(l_d_value)

                wandb.log({"steps": num_steps*config.optim_step_D+s,"D_loss":l_d_value})

            # use the updated D to compute logits_fake
            logits_fake = D(x_fake)
            
            # compute loss for G   
            if(args.loss_type=='gan'):         
                l_g = gan.loss_G(logits_fake)
            else:
                l_g = gan.wgan_loss_G(logits_fake)
            
            # optimize G
            optimizer_G.zero_grad()
            l_g.backward()
            optimizer_G.step()

            l_g_value = l_g.detach().cpu().item() 
            G_loss.append(l_g_value)            
            wandb.log({"steps": num_steps,"G_loss":l_g_value})            
    
            num_steps += 1
            
            # sample the generator
            if num_steps % step_to_sampling == 0:
                with torch.no_grad():
                    G.eval()
                    z = torch.randn(config.batch_size, Dim).to(device)
                    x_sampled = 0.5 * (G(z) + 1.0)
                    im_name = '%s/sampled_image_at_%04d.png' % (result_dir, num_steps)
                    torchvision.utils.save_image(x_sampled, im_name, nrow=16)
                    im_label = "sample at %d" % (num_steps)
                    wandb.log({"steps": num_steps, im_label: wandb.Image(im_name)})
                    G.train()

            tq.update(config.batch_size)
            epoch_postfix = "G_loss=%.5f, D_loss=%.5f" % (l_g_value, l_d_value)
            tq.set_postfix_str(epoch_postfix)
            
        scheduler_G.step()
        scheduler_D.step()        
        tq.close() 
        
    # create a final sample and save
    G.eval()    
    with torch.no_grad():        
        z = torch.randn(config.batch_size, Dim).to(device)
        x_sampled = 0.5 * (G(z) + 1.0)
        im_name = '%s/final_sampled_images.png' % (result_dir)
        torchvision.utils.save_image(x_sampled, im_name, nrow=16)
        wandb.log({"steps": num_steps,"final_sampled_images":wandb.Image(im_name)})
                        
    return G, D, G_loss, D_loss

def main():

    moment = strftime("%Y%m%d_%H%M%S", gmtime())

    if(args.loss_type=='gan'):
        wandb.init(project="A5_Pytorch_gan", config=config_defaults, tags=moment)
    else:
        wandb.init(project="A5_Pytorch_wgan", config=config_defaults, tags=moment)
    wandb.watch_called = False

    # perform training
    model_G, model_D, G_loss, D_loss = run_training()

    # plot Generator and Discriminitor loss
    fig=plt.figure(figsize=[8, 8])
    plt.plot(G_loss, 'b-')
    plt.xlabel('steps', fontsize=20)
    plt.ylabel('G loss', fontsize=20)
    fig.savefig(os.path.join(result_dir, "plot_G_loss.png"), dpi=300)
    
    fig=plt.figure(figsize=[8, 8])
    plt.plot(D_loss, 'b-')
    plt.xlabel('steps', fontsize=20)
    plt.ylabel('D loss', fontsize=20)
    fig.savefig(os.path.join(result_dir, "plot_D_loss.png"), dpi=300)

if __name__ == '__main__':
    main()