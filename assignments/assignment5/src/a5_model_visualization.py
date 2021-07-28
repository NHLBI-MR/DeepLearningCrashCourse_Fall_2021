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
        default="A3_Pytorch_unet_model_20210728_011431.pt",
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
data_dir = os.path.join(Project_DIR, "../data/unet_test_images")
mask_dir = os.path.join(Project_DIR, "../data/unet_test_masks")

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
    
test_samples = np.array(test_samples)
test_masks = np.array(test_masks)
                            
# declare the loss
loss_func = unet_model.LossBinarySegmentation()

device = util.find_GPU()

# ----------------------------------
def run_model_loading():
    """Load pre-trained model

    Returns:
        model : a unet model with all weights loaded
    """

    # get the sample size    
    B, C, H, W = test_samples.shape
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
        images ([N, C, H, W]) : numpy array, samples to test
        masks ([N, 1, H, W]) : numpy array, masks for samples
        loss : loss function
        
    Returns:
        s_maps ([N, C, H, W]): saliency maps
    
    """
        
    # *** START CODE HERE ***
    # for every sample in test_samples, compute saliency map
       
    x = torch.from_numpy(images).type(torch.float32)
    y = torch.from_numpy(masks).type(torch.float32)  
                                
    x = x.to(device=device)
    y = y.to(device=device)

    model.eval()
    x.requires_grad = True
    y.requires_grad = False
    scores = model(x)
    L = loss(scores, y)
    L.backward()

    s_maps = x.grad.data
    s_maps = s_maps.detach().cpu().numpy()
       
    return s_maps
    # *** END CODE HERE ***
    
def compute_smoothing_grad_saliency_maps(model, images, masks, loss, sigma=0.01, num_rep=20):
    """Compute saliency map for images
    
    Args:     
        model : pre-loaded pytorch model
        images ([N, C, H, W]) : numpy array, samples to test
        masks ([N, 1, H, W]) : numpy array, masks for samples
        loss : loss function
        sigma : the noise level
        num_rep : number of repetitions
        
    Returns:
        s_maps ([N, C, H, W]): saliency maps
    
    """
        
    # *** START CODE HERE ***
    # for every sample in test_samples, compute saliency map with SmoothingGrad method

    model.eval()    
    
    s_maps = np.zeros(images.shape)
    
    for n in range(num_rep):
        n = np.random.randn(*images.shape)
        images_with_noise = images + sigma*n 
        s_maps += compute_saliency_map(model, images_with_noise, masks, loss)        
        
    s_maps /= num_rep

    return s_maps
    # *** END CODE HERE ***
 
def main():
    
    # load the pytorch model from saved weights
    model = run_model_loading()        
    model.to(device=device)

    s_maps = compute_saliency_map(model, test_samples, test_masks, loss_func)
    print("Compute the salicency map -- completed")
       
    columns = 4  
    map_to_plot = 10*np.abs(s_maps)/np.max(s_maps)
    f = util.plot_image_array(np.transpose(map_to_plot, (2,3,1,0)), columns=columns)
    fname = os.path.join(result_dir, "salicency_map.png")
    print("Saliceny map is saved to ", fname)
    f.savefig(fname, dpi=300)

    # ---------------------------

    s_maps_smoothing_grad = compute_smoothing_grad_saliency_maps(model, test_samples, test_masks, loss_func)
    print("Compute the SmoothingGrad salicency map -- completed")
    
    map_to_plot = 10*np.abs(s_maps_smoothing_grad)/np.max(s_maps_smoothing_grad)
    f = util.plot_image_array(np.transpose(map_to_plot, (2,3,1,0)), columns=columns)
    fname = os.path.join(result_dir, "salicency_map_smoothing_grad.png")
    print("SmoothingGrad saliceny map is saved to ", fname)
    f.savefig(fname, dpi=300)
    
if __name__ == '__main__':
    main()