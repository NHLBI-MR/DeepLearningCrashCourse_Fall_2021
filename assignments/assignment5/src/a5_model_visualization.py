##################################################
## Deep learning crash course, assignment 5
##################################################
## Description : model visualization from pre-trained model
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
import matplotlib
import matplotlib.pyplot as plt
import argparse
import time

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

import util
import unet_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
   
# disable the interactive plotting
matplotlib.use("agg")

# ----------------------------------
def add_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pytorch Model Visualization")

    parser.add_argument(
        "--model_file",
        type=str,
        default="a5_unet_model.pt",
        help='model file to load')
           
    return parser
# ----------------------------------

# load parameters
args = add_args().parse_args()
print(args)

# ----------------------------------

result_dir = os.path.join(Project_DIR, "../result/model_visual")
os.makedirs(result_dir, exist_ok=True)

model_dir = os.path.join(Project_DIR, "../model")
data_dir = os.path.join(Project_DIR, "../data/unet_test_samples")
mask_dir = os.path.join(Project_DIR, "../data/unet_test_sample_masks")

# find the list of testing files in data_dir
test_samples = []
test_masks = []
image_files = os.listdir(data_dir)
for im_file in image_files:
    if(im_file.find('.npy')!=-1):        
        mask_file_name = os.path.join(mask_dir, im_file)
        if(os.path.isfile(mask_file_name)):
            im = np.load(os.path.join(data_dir, im_file))
            mask = np.load(mask_file_name)

            test_samples.append(np.transpose(im, (2, 0, 1)))
            test_masks.append(np.expand_dims(mask, axis=0))
                        
test_samples = [im/np.max(im) for im in test_samples]
                        
# declare the loss
loss_func = unet_model.LossBinarySegmentation()

# ----------------------------------
def run_model_loading():
    """Load pre-trained model

    Returns:
        model : a unet model with all weights loaded
    """

    # get the sample size
    im = np.load(os.join(data_dir, test_samples[0]))
    C, H, W = im.shape
    print('test image has the shape [%d, %d, %d]' % (H, W, C))

    # *** START CODE HERE ***
    # load the model weights to a unet_model
        
    # declear the model
    model = unet_model.UNet(H, W, C)
    print(model)

    # load the model
    model_full_name = os.path.join(model_dir, args.model_file)
    record = torch.load(model_full_name)
    print("Load saved model ", model_full_name)
    model.load_state_dict(record['model'])
    # *** END CODE HERE ***
    
    return model

def compute_saliency_map(model, images, masks, loss):
    """Compute saliency map for images
    
    Args:     
        model : pre-loaded pytorch model
        images ([N, C, H, W]) : samples for inference 
        masks ([N, 1, H, W]) : masks for samples
        loss : loss function
        
    Returns:
        s_maps ([N, C, H, W]): saliency maps
    
    """
        
    # *** START CODE HERE ***
    # for every sample in test_samples, compute saliency map
    
    model.eval()
    x = Variable(images, requires_grad=True)
    scores = model(x)
    loss = loss(scores, masks)
    loss.backward()

    s_maps = x.grad.data
    s_maps = s_maps.detach().numpy()
   
    return s_maps
    # *** END CODE HERE ***
    
def compute_smoothing_grad_saliency_maps(model, images, masks, loss, sigma=0.01, num_rep=10):
    """Compute saliency map for images
    
    Args:     
        model : pre-loaded pytorch model
        images ([N, C, H, W]) : samples for inference 
        masks ([N, 1, H, W]) : masks for samples
        loss : loss function
        sigma : the noise level
        num_rep : number of repetitions
        
    Returns:
        s_maps ([N, C, H, W]): saliency maps
    
    """
        
    # *** START CODE HERE ***
    # for every sample in test_samples, compute saliency map with SmoothingGrad method

    model.eval()    
    
    s_maps = np.zeros_like(images))
    
    for n in range(num_rep):
        n = torch.rand(images.shape)        
        s_maps += compute_saliency_map(model, images+n, masks, loss)        

    s_maps / num_rep

    return s_maps
    # *** END CODE HERE ***
 
def main():
    
    # load the pytorch model from saved weights
    model = run_model_loading()    

    s_maps = compute_saliency_map(model, test_samples, test_masks, loss_func)
    s_maps_smoothing_grad = compute_smoothing_grad_saliency_maps(model, test_samples, test_masks, loss_func)
    
    # plot results
    columns = 4
    figsize=[32, 32]
    
    f = util.plot_image_array(np.transpose(np.abs(s_maps), (2,3,1,0)), columns=columns, figsize=figsize)
    f.savefig(os.path.join(result_dir, "salicency_map.png"), dpi=300)
    
    f = util.plot_image_array(np.transpose(np.abs(s_maps_smoothing_grad), (2,3,1,0)), columns=columns, figsize=figsize)
    f.savefig(os.path.join(result_dir, "salicency_map_smoothing_grad.png"), dpi=300)

if __name__ == '__main__':
    main()