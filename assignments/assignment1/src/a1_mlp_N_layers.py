##################################################
## Deep learning crash course, assignment 1
##################################################
## Description : the N-layer MLP model
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Mmaintainer: xueh2 @ github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

from util import *
# NOTE: get_weight_bias_key and get_score_activation_key may be useful here

def initialize_params(num_input_layer, num_hidden_layers, num_output):
    """
    Initialize the model parameters.

    Weights are initialized with a random normal distribution.
    Biases are initialized with zero.

    Inputs:
        num_input_layer : number of neurons in the input layer
        num_hidden_layers : number of neurons for each hidden layer, a list, e.g. [300, 100, 50]
        num_output : number of neurons in the output layer

    Outputs:
        params : a dictionary storing weights and biases, 
        params['W1'], params['b1'], params['W2'], params['b2'], params['W2'], params['b3'], ...
    """

    # *** START CODE HERE ***
    params = dict()

    input_dim = num_input_layer

    N = len(num_hidden_layers)

    for layer in range(N):
        weight_key, bias_key = get_weight_bias_key(layer+1)
        if(layer>0):
            input_dim = num_hidden_layers[layer-1]

        params[weight_key] = np.random.randn(input_dim, num_hidden_layers[layer])
        params[bias_key] = np.zeros((1, num_hidden_layers[layer]))

    weight_key, bias_key = get_weight_bias_key(N+1)
    params[weight_key] = np.random.randn(num_hidden_layers[N-1], num_output)
    params[bias_key] = np.zeros((1, num_output))

    params['num_hidden_layers'] = num_hidden_layers

    return params
    # *** END CODE HERE ***

def forward_pass(data, labels, params):
    """
    Implement the forward pass.

    Inputs:
        data: [B, D] a mini-batch of samples
        labels: [B, D], one-hot encoded labels for all samples
        params: A dictionary contains W1, b1, W2, b2, W3 and b3

    Outputs:
        loss : the computed loss, a scalar
        y_hat : [B, C], the probabilities for every class, every sample
        params : You need to buffer the intermediate results for backprog. Store them in this dictionary
    """
    # *** START CODE HERE ***
    B, K = data.shape

    num_hidden_layers = params['num_hidden_layers']
    N = len(num_hidden_layers)

    x = data
    for layer in range(N):
        weight_key, bias_key = get_weight_bias_key(layer+1)
        z = x.dot(params[weight_key]) + params[bias_key]
        a = sigmoid(z)

        score_key, activation_key = get_score_activation_key(layer+1)
        params[score_key] = z
        params[activation_key] = a

        x = a

    weight_key, bias_key = get_weight_bias_key(N+1)
    z = x.dot(params[weight_key]) + params[bias_key]
    y_hat = softmax(z)

    score_key, activation_key = get_score_activation_key(N+1)
    params[score_key] = z
    params[activation_key] = y_hat
    params['y_hat'] = y_hat

    loss = -np.sum(np.multiply(labels, np.log(y_hat + 1e-8))) / B

    return loss, y_hat, params
    # *** END CODE HERE ***

def backward_pass(data, labels, params, reg):
    """
    Implement the backprop of the model and compute the gradient for all weights and biases.

    Inputs:
        data: [B, D] a mini-batch of samples
        labels: [B, D], one-hot encoded labels for all samples
        params: A dictionary contains weights and biases and intermediate results
        reg : regularization strength

    Outputs:
        grad : a dictionary contains gradients for weights and biases
        grad['W1'], grad['b1'], grad['W2'], grad['b2'], grad['W3'], grad['b3'] ...

    You need to compute the gradient of loss to all weights and biases, including taking derivatives through the softmax.
    """
    # *** START CODE HERE ***
    num_hidden_layers = params['num_hidden_layers']
    N = len(num_hidden_layers)
    y_hat = params['y_hat']

    B, K = data.shape

    grad = dict()

    for layer in np.arange(N+1, 0, -1):
        weight_key, bias_key = get_weight_bias_key(layer)
        score_key, activation_key = get_score_activation_key(layer)
        score_key_prev, activation_key_prev = get_score_activation_key(layer-1)

        if(layer==N+1):
            dz = -(labels - y_hat) 
        else:
            dz = np.multiply(grad[activation_key], params[activation_key] * (1-params[activation_key]))

        if(layer>1):
            dW = np.dot(params[activation_key_prev].T, dz)
        else:
            dW = np.dot(data.T, dz)

        db = np.sum(dz, axis=0, keepdims=True)
        da_prev = np.dot(dz, params[weight_key].T)

        grad[activation_key_prev] = da_prev
        grad[score_key] = dz
        grad[weight_key] = dW
        grad[bias_key] = db

    for layer in np.arange(N+1, 0, -1):
        weight_key, bias_key = get_weight_bias_key(layer)
        grad[weight_key] /= B
        grad[bias_key] /= B

    if(reg>0):
        for layer in np.arange(N+1, 0, -1):
            weight_key, bias_key = get_weight_bias_key(layer)
            grad[weight_key] += 2 * reg * params[weight_key]

    return grad
    # *** END CODE HERE ***

def run_trainning(args, data, labels):
    """Run the training

    Inputs:
        args : arguments
        data : training/validation/test images
        labels : training/validation/test labels

    Outputs:
        params : parameters after training
        loss_train, loss_val : loss for every epoch
        accu_train, accu_val : accuracy for every epoch
    """

    train_data = data['train']
    train_labels = labels['train']

    val_data = data['val']
    val_labels = labels['val']

    # get the training parameters
    num_epochs = args.num_epochs
    num_hidden_layers = args.num_hidden_layers
    batch_size = args.batch_size
    reg = args.reg
    learning_rate = args.learning_rate

    (N, dim) = train_data.shape

    if(args.one_batch_training):
        print("Train with only one batch")
        batch_start = np.random.randint(0, N-batch_size, 1, dtype=int)
        train_data = train_data[batch_start[0]:batch_start[0]+batch_size,:]
        train_labels = train_labels[batch_start[0]:batch_start[0]+batch_size,:]
        (N, dim) = train_data.shape

    # *** START CODE HERE ***
    # initialize the parameters
    params = initialize_params(dim, num_hidden_layers, 10)
    # *** END CODE HERE ***

    num_layers = len(num_hidden_layers) + 1
    num_iter = int(N / batch_size)

    loss_train = []
    loss_val = []
    accu_train = []
    accu_val = []

    # train for num_epochs
    for epoch in range(num_epochs):

        # go through all mini-batches for this epoch
        for iter in range(num_iter):

            # *** START CODE HERE ***
            # get a mini-batch
            ind_s = iter*batch_size
            ind_e = ind_s + batch_size
            data = train_data[ind_s:ind_e, :]
            labels = train_labels[ind_s:ind_e, :]

            # forward pass
            loss, y_hat, params = forward_pass(data, labels, params)

            # backprop
            grad = backward_pass(data, labels, params, reg)

            # perform gradient descent step
            for layer in range(num_layers):
                weight_key, bias_key = get_weight_bias_key(layer+1)
                params[weight_key] -= learning_rate * grad[weight_key]
                params[bias_key] -= learning_rate * grad[bias_key]
            # *** END CODE HERE ***

        # after one epoch, compute training loss and accuracy
        train_loss, y_hat, params = forward_pass(train_data, train_labels, params)
        loss_train.append(train_loss)
        train_accu = compute_accuracy(y_hat, train_labels)
        accu_train.append(train_accu)

        # after one epoch, compute validation loss and accuracy
        val_loss, y_hat, params = forward_pass(val_data, val_labels, params)
        loss_val.append(val_loss)
        val_accu = compute_accuracy(y_hat, val_labels)
        accu_val.append(val_accu)

        print('epoch %d, train loss %f, accuracy %f - val loss %f, accuracy %f' % (epoch, train_loss, train_accu, val_loss, val_accu))

    return params, loss_train, loss_val, accu_train, accu_val

def add_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="N-layer MLP for MNIST classsification")

    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument("--num_hidden_layers", type=int, nargs="+", default=[300, 200, 100])
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--reg', type=float, default=0.0, help='regularization lambda')
    parser.add_argument('--learning_rate', type=float, default=1.0, help='learn rate')

    parser.add_argument(
        "--training_record",
        type=str,
        default="base_line_N_layer_MLP",
        help='String to record this training')

    parser.add_argument('--one_batch_training', type=bool, default=False, help='if True, train with only one batch, for debugging purpose')

    return parser

def main():

    # load parameters
    args = add_args().parse_args()
    print(args)

    # make sure results are more reproducible
    np.random.seed(1024)

    # load data
    train_data, train_labels, val_data, val_labels, test_data, test_labels = load_data(data_dir=os.path.join(Project_DIR, '../data'), force_loading=False)

    # perform training
    params, loss_train, loss_val, accu_train, accu_val = run_trainning(
                                                                        args, 
                                                                        {'train':train_data, 'val':val_data, 'test':test_data}, 
                                                                        {'train':train_labels, 'val':val_labels, 'test':test_labels}
                                                                    )

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

    fig.savefig(os.path.join(Project_DIR, '../results/' + args.training_record + '.pdf'))

    # compute arraucy on the test dataset
    test_loss, y_hat, params = forward_pass(test_data, test_labels, params)
    accuracy = compute_accuracy(y_hat, test_labels)
    print('Test accuracy is %f for training run %s' % (accuracy, args.training_record))

if __name__ == '__main__':
    main()