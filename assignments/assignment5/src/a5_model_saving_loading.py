##################################################
## Deep learning crash course, assignment 5
##################################################
## Description : model format conversion, saving and loading
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
from scipy.special import softmax
from scipy.special import expit
import copy

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

import util
import unet_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnxruntime as ort
   
# disable the interactive plotting
matplotlib.use("agg")

# ----------------------------------
def add_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pytorch Model Saving and Conversion")

    parser.add_argument(
        "--model_file",
        type=str,
        default="A3_Pytorch_unet_model_20210728_011431.pt",
        help='model file to load')
    
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        help='format, either "torchscript" or "onnx" ')
        
    return parser
# ----------------------------------

# load parameters
args = add_args().parse_args()
print(args)

# ----------------------------------

result_dir = os.path.join(Project_DIR, "../result/model_saving")
os.makedirs(result_dir, exist_ok=True)

model_dir = os.path.join(Project_DIR, "../model")
data_dir = os.path.join(Project_DIR, "../data/unet_test_images")

# find the list of testing files in data_dir
test_samples = []
image_files = os.listdir(data_dir)
for im_file in image_files:
    if(im_file.find('.npy')!=-1):
        test_samples.append(im_file)

# ----------------------------------
def run_model_loading():
    """Load pre-trained model

    Returns:
        model : a unet model with all weights loaded
    """

    # get the sample size
    im = np.load(os.path.join(data_dir, test_samples[0]))
    H, W, C = im.shape
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

def run_model_conversion(model):
    """Convert model to torch script or onnx format
    
    Args:     
        model : pre-loaded pytorch model
        
    Returns:
        model_output : model in torch script or onnx format
    
    """
    
    im = np.load(os.path.join(data_dir, test_samples[0]))
    H, W, C = im.shape
    
    filename, file_extension = os.path.splitext(args.model_file)
    
    if(args.format == 'onnx'):
        # convert to ONNX format and save onnx model to model_output_name
        model_output_name = os.path.join(result_dir, filename+'.onnx')
        
        # *** START CODE HERE ***
        model.eval()

        x = torch.randn(1, C, H, W, requires_grad=True)
        torch_out = model(x)
        
        torch.onnx.export(model, x, model_output_name, export_params=True, opset_version=11, input_names = ['input'], output_names = ['output'], dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}})
        # *** END CODE HERE ***
    else:
        
        # convert to torch script format and save traced model to model_output_name
        model_output_name = os.path.join(result_dir, filename+'.pts')
        
        # *** START CODE HERE ***
        model_for_tracing = copy.deepcopy((model))
        model_traced = torch.jit.trace(model_for_tracing, (torch.rand(1, C, H, W)))
        model_traced.save(model_output_name)
        # *** END CODE HERE ***
    
def run_model_inference(model):
    """Run the model inference

    Args:
        model : pytorch model
    """

    # read in the test samples
    N = len(test_samples)
    im = np.load(os.path.join(data_dir, test_samples[0]))
    H, W, C = im.shape
    
    filename, file_extension = os.path.splitext(args.model_file)
    
    images = np.zeros((N, C, H, W), dtype=np.float32)
    for n, sample in enumerate(test_samples):
        im = np.load(os.path.join(data_dir, sample))
        images[n, :, :, :] = np.transpose(im, (2, 0, 1)) / np.max(im)

    # *** START CODE HERE ***
    # run the original model inference on images and save model outputs in scores
    x = torch.from_numpy(images).type(torch.float32)
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        scores = model(x)        
    t1 = time.time()
    print("Pytorch model runs in %.2f seconds " % (t1 - t0))
    # *** END CODE HERE ***
    
    # convert logits to probability
    probs = torch.sigmoid(scores)
    
    # plot and save the testing samples
    columns = 4
    figsize=[32, 32]
    
    f = util.plot_image_array(np.transpose(images, (2,3,1,0)), columns=columns, figsize=figsize)
    f.savefig(os.path.join(result_dir, "test_samples.png"), dpi=300)
    
    f = util.plot_image_array(np.transpose(probs.numpy(), (2,3,1,0)), columns=columns, figsize=figsize)
    f.savefig(os.path.join(result_dir, "test_results_for_pytorch_model.png"), dpi=300)
        
    if(args.format == 'onnx'):
        
        model_onnx_name = os.path.join(result_dir, filename+'.onnx')
        
        t0 = time.time()
        # *** START CODE HERE ***
        # load the saved onnx model from model_onnx_name
        model_onnx = ort.InferenceSession(model_onnx_name)

        # perform model inference on images and save the probability into probs_onnx        
        input_name = model_onnx.get_inputs()[0].name
        output_name = model_onnx.get_outputs()[0].name
        y_pred = model_onnx.run([output_name], {input_name: images})[0]
        probs_onnx = expit(y_pred)        
        # *** END CODE HERE ***
        t1 = time.time()
        print("Onnx model inference in %.2f seconds " % (t1 - t0))
        
        # check whether onnx model is working correctly
        diff = torch.norm(probs - torch.from_numpy(probs_onnx))
        print("Onnx test run, diff is %f" % diff)

        # plot and save onnx model inference results
        f = util.plot_image_array(np.transpose(probs_onnx, (2,3,1,0)), columns=columns, figsize=figsize)
        f.savefig(os.path.join(result_dir, "test_results_for_saved_onnx_model.png"), dpi=300)
    else:
        
        model_pts_name = os.path.join(result_dir, filename+'.pts')

        t0 = time.time()
        # *** START CODE HERE ***        
        # load the saved torch script model from model_pts_name
        model_traced = torch.jit.load(model_pts_name)
        x = torch.from_numpy(images).type(torch.float32)
        with torch.no_grad():
            scores = model_traced(x)
            probs_traced = torch.sigmoid(scores)
        
        # *** END CODE HERE ***
        t1 = time.time()
        print("Traced model inference in %.2f seconds " % (t1 - t0))
        
        # check whether script model is working correctly
        diff = torch.norm(probs - probs_traced)
        print("Torch script test run, diff is %f" % diff)
        
        # plot and save torch script model inference results
        f = util.plot_image_array(np.transpose(probs_traced, (2,3,1,0)), columns=columns, figsize=figsize)
        f.savefig(os.path.join(result_dir, "test_results_for_saved_torchscript_model.png"), dpi=300)       
 
def main():
    
    # load the pytorch model from saved weights
    model_pytorch = run_model_loading()    
    # convert model
    run_model_conversion(model_pytorch)
    
    # test the saved the model
    run_model_inference(model_pytorch)

if __name__ == '__main__':
    main()