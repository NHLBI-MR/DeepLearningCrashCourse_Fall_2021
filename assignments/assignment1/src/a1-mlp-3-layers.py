import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

from util import *

def initialize_params(input_size, num_hidden1, num_hidden2, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer
    b1 is the bias vector for the hidden layer
    W2 is the weight matrix for the output layers
    b2 is the bias vector for the output layer

    As specified in the PDF, weight matricesshould be initialized with a random normal distribution.
    Bias vectors should be initialized with zero.
    
    Args:
        input_size: The size of the input data dimension (784)
        num_hidden: The number of hidden states
        num_output: The number of output classes
    
    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # *** START CODE HERE ***
    params = dict()
    params['W1'] = np.random.randn(input_size, num_hidden1)
    params['b1'] = np.zeros((1, num_hidden1))
    params['W2'] = np.random.randn(num_hidden1, num_output)
    params['b2'] = np.zeros((1, num_output))
    return params
    # *** END CODE HERE ***

def forward_prop(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.
    
    Args:
        data: A numpy array containing the input
        labels: A 1d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    B, K = data.shape
    
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    
    z1 = data.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    y_hat = softmax(z2)

    loss = -np.sum(np.multiply(labels, np.log(y_hat + 1e-8))) / B
    return a1, y_hat, loss
    # *** END CODE HERE ***

def backward_prop2(data, labels, params, a1, y_hat, reg):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 1d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    B, K = data.shape
    
    #a1, y_hat, loss = forward_prop(data, labels, params)
    
    dz2 = -(labels - y_hat) 
    dW2 = np.dot(a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)
    
    da1 = np.dot(dz2, W2.T)
    dz1 = np.multiply(da1, a1 * (1-a1))
    
    dW1 = np.dot(data.T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)
    # *** END CODE HERE ***
    
    grad = dict()
    grad['W1'] = dW1 / B
    grad['b1'] = db1 / B
    grad['W2'] = dW2 / B
    grad['b2'] = db2 / B

    if(reg>0):
        grad['W1'] += 2 * reg * W1
        grad['W2'] += 2 * reg * W2

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
    num_hidden1 = args.num_hidden1
    num_hidden2 = args.num_hidden2
    batch_size = args.batch_size
    reg = args.reg
    learning_rate = args.learning_rate

    (N, dim) = train_data.shape

    # initialize the parameters
    params = initialize_params(dim, num_hidden1, num_hidden2, 10)

    num_iter = int(N / batch_size)

    loss_train = []
    loss_val = []
    accu_train = []
    accu_val = []

    # train for num_epochs
    for epoch in range(num_epochs):

        # go through all mini-batches for this epoch
        for iter in range(num_iter):

            # get a mini-batch
            ind_s = iter*batch_size
            ind_e = ind_s + batch_size
            data = train_data[ind_s:ind_e, :]
            labels = train_labels[ind_s:ind_e, :]

            # forward pass
            a1, y_hat, loss = forward_prop(data, labels, params)

            # backprop
            grad = backward_prop2(data, labels, params, a1, y_hat, reg)

            # perform gradient descent step
            params['W1'] -= learning_rate * grad['W1']
            params['b1'] -= learning_rate * grad['b1']
            params['W2'] -= learning_rate * grad['W2']
            params['b2'] -= learning_rate * grad['b2']

        # after one epoch, compute training loss and accuracy
        h, output, train_loss = forward_prop(train_data, train_labels, params)
        loss_train.append(train_loss)
        train_accu = compute_accuracy(output, train_labels)
        accu_train.append(train_accu)

        # after one epoch, compute validation loss and accuracy
        h, output, val_loss = forward_prop(val_data, val_labels, params)
        loss_val.append(val_loss)
        val_accu = compute_accuracy(output, val_labels)
        accu_val.append(val_accu)

        print('epoch %d, train loss %f, accuracy %f - val loss %f, accuracy %f' % (epoch, train_loss, train_accu, val_loss, val_accu))

    return params, loss_train, loss_val, accu_train, accu_val

def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="MLP for MNIST classsification")

    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--num_hidden1', type=int, default=300, help='size for the first hidden layer')
    parser.add_argument('--num_hidden2', type=int, default=300, help='size for the second hidden layer')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--reg', type=float, default=0.0, help='regularization lambda')
    parser.add_argument('--learning_rate', type=float, default=5.0, help='learn rate')

    parser.add_argument(
        "--training_record",
        type=str,
        default="base_line",
        help='String to record this training')

    args = parser.parse_args()
    return args

def main():

    # load parameters
    args = _parse_args()
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
    ax2.set_title(args.training_record)
    ax2.legend()

    fig.savefig(os.path.join(Project_DIR, '../results/' + args.training_record + '.pdf'))

    # compute arraucy on the test dataset
    h, output, cost = forward_prop(test_data, test_labels, params)
    accuracy = compute_accuracy(output, test_labels)
    print('Test accuracy is %f for training run %s' % (accuracy, args.training_record))

if __name__ == '__main__':
    main()
