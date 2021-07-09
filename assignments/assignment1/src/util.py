import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

# number of samples for validation, as the MNIST convention
NUM_VALIDATION=10000

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == 
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def load_data(data_dir='../data', force_loading=False):
    """
    Load train and test data

    Outputs : train_data, train_label, val_data, val_label, test_data, test_label

    train_data is a [N, D] array, every row is an image
    train_label is a [N, D] array, one-hot encoded image label

    val_data, val_label are for the validation set
    test_data, test_label are for the test set

    This function will save the loaded train/val/test data to speedup data loading
    """

    # if the buffer does not exist, always loading the data

    train_buf_file = os.path.join(data_dir, 'train.dat')
    val_buf_file = os.path.join(data_dir, 'val.dat')
    test_buf_file = os.path.join(data_dir, 'test.dat')

    require_to_load = False
    if(not os.path.isfile(train_buf_file) or not os.path.isfile(test_buf_file) or not os.path.isfile(val_buf_file)):
        require_to_load = True

    if(force_loading or require_to_load):
        print(f'Load image and label from csf files ...')

        # load data and save the buffer
        train_image_file = os.path.join(data_dir, 'images_train.csv')
        train_label_file = os.path.join(data_dir, 'labels_train.csv')

        train_data, train_labels = read_data(train_image_file, train_label_file)
        train_labels = one_hot_labels(train_labels)
        data_index = np.random.permutation(train_data.shape[0])
        train_data = train_data[data_index,:]
        train_labels = train_labels[data_index,:]

        val_data = train_data[0:NUM_VALIDATION,:]
        val_labels = train_labels[0:NUM_VALIDATION,:]
        train_data = train_data[NUM_VALIDATION:,:]
        train_labels = train_labels[NUM_VALIDATION:,:]
        mean = np.mean(train_data)
        std = np.std(train_data)
        train_data = (train_data - mean) / std
        val_data = (val_data - mean) / std

        test_image_file = os.path.join(data_dir, 'images_test.csv')
        test_label_file = os.path.join(data_dir, 'labels_test.csv')
        test_data, test_labels = read_data(test_image_file, test_label_file)
        test_labels = one_hot_labels(test_labels)
        test_data = (test_data - mean) / std

        # save the buffer
        pickle.dump((train_data, train_labels), open(train_buf_file, 'wb'))
        pickle.dump((val_data, val_labels), open(val_buf_file, 'wb'))
        pickle.dump((test_data, test_labels), open(test_buf_file, 'wb'))
    else:
        print(f'Load image and label from pre-saved buffer ...')
        train_data, train_labels = pickle.load( open(train_buf_file, 'rb'))
        val_data, val_labels = pickle.load( open(val_buf_file, 'rb'))
        test_data, test_labels = pickle.load( open(test_buf_file, 'rb'))

    return train_data, train_labels, val_data, val_labels, test_data, test_labels

def softmax(x):
    """
    Softmax function, using the numericla stable formula.

    Inputs:
        x: [N, C], N samples, C classes

    Outputs:
        s: [N, C], results after softmax along axis=1 in x
    """

    v = np.exp(x - np.max(x, axis=1, keepdims=True))
    s = v / np.sum(v, axis=1, keepdims=True)
    return s

def sigmoid(x):
    """
    The element-wise sigmoid function.

    Inputs:
        x: numpy array

    Outputs:
        sigmoid(x)
    """
    z = 1 / (1 + np.exp(-x))
    return z
