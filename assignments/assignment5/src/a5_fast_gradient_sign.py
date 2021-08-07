##################################################
## Deep learning crash course, assignment 5
##################################################
## Description : fast gradient sign attack
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Maintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

import torch

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

util.set_seed()

# ----------------------------------
def add_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pytorch CNN model for Cifar10 classification, using wandb")

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.002, 0.004, 0.006, 0.008, 0.1, 0.2])
    
    parser.add_argument(
        "--model_file",
        type=str,
        default="A3_Pytorch_small_resnet_20210728_175702.pt",
        help='Pre-trained model')

    return parser
# ----------------------------------
   
# load dataset
data_dir = os.path.join(Project_DIR, "../data/cifar10")
cifar10_dataset = util.load_and_prepare_data(data_dir, subtract_mean=True)

# load parameters
args = add_args().parse_args()
print(args)

result_dir = os.path.join(Project_DIR, "../result/fast_gradient_sign_attack")
os.makedirs(result_dir, exist_ok=True)

model_dir = os.path.join(Project_DIR, "../model")

# ----------------------------------
     
def run_model_loading():
    """Load pre-trained model

    Returns:
        model : a resnet model for cifar 10 dataset
    """

    # *** START CODE HERE ***
    # load the model weights to a unet_model
    # get the sample size
    H, W, C, B = cifar10_dataset['X_train'].shape

    # declare the model m
    model = resnet_model.Cifar10SmallResNet(H, W, C)
    print(model)

    # load the model
    model_full_name = os.path.join(model_dir, args.model_file)
    record = torch.load(model_full_name)
    print("Load saved model ", model_full_name)
    model.load_state_dict(record['model'])
    # *** END CODE HERE ***
    
    return model
   
def compute_adverserial_examples(x, dx, epsilon):
    """Compute FSGM adverserial examples

    Args:
        x ([B, 3, H, W]): a batch of test samples
        dx ([B, 3, H, W]): grad of x
        epsilon (float): strength of purturbation
        
    Returns:
        x_adv : adverserial examples
    """
    # *** START CODE HERE ***
    # return the adverserial examples
    x_adv = x + epsilon * dx.sign()
    x_adv = torch.clamp(x_adv, -1.0, 1.0)
    return x_adv
    # *** END CODE HERE ***
   
def run_fast_gradient_sign_attack(model, epsilon, loader_for_test, device):
    """Perform FGS attack on test samples

    Args:
        model : pre-trained model

    Returns:
        accu : accuracy for test samples
        accu_adv : accuracy for adverserial examples
        test_images : test images
        test_images_adv : test images with perturbation
        test_labels : test labels
        test_labels_adv : test labels after perturbation
    """
    
    # declare the loss function, loss_func
    loss_func = nn.CrossEntropyLoss()

    # perform attacking on test samples
    correct = 0
    correct_adv = 0
    
    test_images = []
    test_labels = []
    test_images_adv = []
    test_labels_adv = []
    
    model.to(device=device)    
    model.eval()
    for i, data in enumerate(loader_for_test, 0):
        x, y = data
        
        test_images.append(x)
        test_labels.append(y)
        
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.long)

        x.requires_grad = True

        # without perturbation
        y_hat = model(x)
        loss = loss_func(y_hat, y)

        # original accuracy
        correct += util.compute_accuracy(y_hat.detach().cpu(), y.detach().cpu())
        
        # do the perturbation
        model.zero_grad()
        loss.backward()

        if(epsilon>0):
            # *** START CODE HERE ***
            # get the gradient of loss to x, compute adverserial examples x_adv, compute the accuracy for this adverserial batch
            
            # gradients
            dx = x.grad.data
            
            # call fsgm
            x_adv = compute_adverserial_examples(x, dx, epsilon)
            
            # compute the accuracy on x_adv
            y_hat_adv = model(x_adv)
            correct_adv += util.compute_accuracy(y_hat_adv.detach().cpu(), y.detach().cpu())
            
            _, pred = torch.max(y_hat_adv, 1)
            # *** END CODE HERE ***
        else:
            x_adv = x
            correct_adv = correct
            _, pred = torch.max(y_hat, 1)

        test_images_adv.append(x_adv.detach().cpu())
        test_labels_adv.append(pred.detach().cpu())
        
    accu = correct / len(loader_for_test)
    accu_adv = correct_adv / len(loader_for_test)

    return accu, accu_adv, test_images, test_images_adv, test_labels, test_labels_adv

def main():

    # load the saved model
    model = run_model_loading()

    # get the device    
    device = util.find_GPU()
    
    # prepare the datasets and loaders
    train_set, test_set, loader_for_train, loader_for_val, loader_for_test = set_up_cifar_10_dataset(cifar10_dataset, num_samples_validation=1000, batch_size=args.batch_size)

    with open(os.path.join(data_dir, "batches.meta"), "rb") as f:
        label_names = pickle.load(f, encoding="latin1")
        
    # perform attack
    accuracies = []

    columns=8
    figsize=[16, 16]    
    
    for epsilon in args.epsilons:
        accu, accu_adv, test_images, test_images_adv, test_labels, test_labels_adv = run_fast_gradient_sign_attack(model, epsilon, loader_for_test, device)
        accuracies.append(accu_adv)

        print("epsilon = %.4f, accuracy = %.4f" % (epsilon, accu_adv))

        ind = np.random.randint(0, len(test_images_adv))
        im = np.transpose(test_images_adv[ind].numpy(), (2, 3, 1, 0))
        im += cifar10_dataset['mean_image']
        
        f = plot_adverserial_attack_image_array(im, test_labels[ind].numpy(), label_names['label_names'], test_labels_adv[ind].numpy(), columns=columns, figsize=figsize)
        f.savefig(os.path.join(result_dir, "adverserial_examples_epsilon%.4f.png" % epsilon), dpi=100)
        
    # plot accuracy
    fig=plt.figure(figsize=[8, 8])
    plt.plot(args.epsilons, accuracies, 'b+')
    plt.plot(args.epsilons, accuracies, 'b-')
    plt.xlabel('epsilons', fontsize=20)
    plt.ylabel('accuracy', fontsize=20)
    fig.savefig(os.path.join(result_dir, "plot_epsilons_vs_accuracy.png"), dpi=300)
    
if __name__ == '__main__':
    main()