##################################################
## Deep learning crash course, assignment 1
##################################################
## Description : the three-layer MLP model
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Maintainer: xueh2 @ github
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

set_seed()

def initialize_params(num_input_layer, num_hidden1, num_hidden2, num_output):
    """
    Initialize the model parameters.

    Weights are initialized with a random normal distribution.
    Biases are initialized with zero.

    Inputs:
        num_input_layer : number of neurons in the input layer
        num_hidden1 : number of neurons in the first hidden layer
        num_hidden2 : number of neurons in the second hidden layer
        num_output : number of neurons in the output layer

    Outputs:
        params : a dictionary storing weights and biases, 
        params['W1'], params['b1'], params['W2'], params['b2'], params['W2'], params['b3']
    """

    # *** START CODE HERE ***
    params = dict()
    params['W1'] = np.random.randn(num_input_layer, num_hidden1)
    params['b1'] = np.zeros((1, num_hidden1))
    params['W2'] = np.random.randn(num_hidden1, num_hidden2)
    params['b2'] = np.zeros((1, num_hidden2))
    params['W3'] = np.random.randn(num_hidden2, num_output)
    params['b3'] = np.zeros((1, num_output))
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

    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    W3 = params['W3']
    b3 = params['b3']

    z1 = data.dot(W1) + b1
    a1 = sigmoid(z1)

    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)

    z3 = a2.dot(W3) + b3
    y_hat = softmax(z3)

    loss = -np.sum(np.multiply(labels, np.log(y_hat + 1e-8))) / B

    params['z1'] = z1
    params['a1'] = a1
    params['z2'] = z2
    params['a2'] = a2
    params['z3'] = z3
    params['y_hat'] = y_hat
    
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
        grad['W1'], grad['b1'], grad['W2'], grad['b2'], grad['W3'], grad['b3'] 

    You need to compute the gradient of loss to all weights and biases, including taking derivatives through the softmax.
    """
    # *** START CODE HERE ***
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    W3 = params['W3']
    b3 = params['b3']

    z1 = params['z1']
    a1 = params['a1']
    z2 = params['z2']
    a2 = params['a2']
    z3 = params['z3']
    y_hat = params['y_hat']

    B, K = data.shape

    dz3 = -(labels - y_hat) 
    dW3 = np.dot(a2.T, dz3)
    db3 = np.sum(dz3, axis=0, keepdims=True)

    da2 = np.dot(dz3, W3.T)
    dz2 = np.multiply(da2, a2 * (1-a2))
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = np.dot(dz2, W2.T)
    dz1 = np.multiply(da1, a1 * (1-a1))
    dW1 = np.dot(data.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    grad = dict()
    grad['W1'] = dW1 / B
    grad['b1'] = db1 / B
    grad['W2'] = dW2 / B
    grad['b2'] = db2 / B
    grad['W3'] = dW3 / B
    grad['b3'] = db3 / B

    if(reg>0):
        grad['W1'] += 2 * reg * W1
        grad['W2'] += 2 * reg * W2
        grad['W3'] += 2 * reg * W3

    return grad
    # *** END CODE HERE ***

def run_training(args, data, labels):
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
    num_hidden1 = args.num_hidden1
    num_hidden2 = args.num_hidden2
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
    params = initialize_params(dim, num_hidden1, num_hidden2, 10)
    # *** END CODE HERE ***

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
            params['W1'] -= learning_rate * grad['W1']
            params['b1'] -= learning_rate * grad['b1']
            params['W2'] -= learning_rate * grad['W2']
            params['b2'] -= learning_rate * grad['b2']
            params['W3'] -= learning_rate * grad['W3']
            params['b3'] -= learning_rate * grad['b3']
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
    parser = argparse.ArgumentParser(description="Three-layer MLP for MNIST classsification")

    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--num_hidden1', type=int, default=200, help='size for the first hidden layer')
    parser.add_argument('--num_hidden2', type=int, default=100, help='size for the second hidden layer')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--reg', type=float, default=0.0, help='regularization lambda')
    parser.add_argument('--learning_rate', type=float, default=5.0, help='learn rate')

    parser.add_argument(
        "--training_record",
        type=str,
        default="base_line",
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
    params, loss_train, loss_val, accu_train, accu_val = run_training(
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
    ax2.set_title(args.training_record)
    ax2.legend()

    res_dir = os.path.join(Project_DIR, '../results')
    os.makedirs(res_dir, exist_ok=True)
    fig.savefig(os.path.join(res_dir, args.training_record + '.pdf'))

    # compute arraucy on the test dataset
    test_loss, y_hat, params = forward_pass(test_data, test_labels, params)
    accuracy = compute_accuracy(y_hat, test_labels)
    print('Test accuracy is %f for training run %s' % (accuracy, args.training_record))

if __name__ == '__main__':
    main()