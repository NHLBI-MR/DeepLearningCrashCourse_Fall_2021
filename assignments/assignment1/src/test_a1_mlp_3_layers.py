##################################################
## Deep learning cash course, assignment 1
##################################################
## Description : unit test for the three-layer MLP model
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Mmaintainer: xueh2 @ github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

import time
import os
import sys
from pathlib import Path
import numpy as np

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

import unittest
from util import *
from a1_mlp_3_layers import * 

class Test_Three_Layer_MLP(unittest.TestCase):
    """Tests for the Three-layer MLP."""

    def setUp(self):

        parser = add_args()
        self.args = parser.parse_args(["--num_epochs", "30", "--num_hidden1", "200", "--num_hidden2", "100", "--batch_size", "1000", "--reg", "0.0001", "--learning_rate", "5.0", "--training_record", "unittest"])
        print("Test_Three_Layer_MLP, args ", self.args)

        # load the test data
        np.random.seed(1024)

        # load data
        self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = load_data(data_dir=os.path.join(Project_DIR, '../data'), force_loading=False)

    def tearDown(self):
        pass

    def test_three_layer_mlp(self):
        """Test the three-layer MLP model by performing a training and checking accuracy on test set."""

        # *** START CODE HERE ***
        # perform training
        t0 = time.time()
        params, loss_train, loss_val, accu_train, accu_val = run_trainning(
                                                                        self.args, 
                                                                        {'train':self.train_data, 'val':self.val_data, 'test':self.test_data}, 
                                                                        {'train':self.train_labels, 'val':self.val_labels, 'test':self.test_labels}
                                                                    )
        t1 = time.time()
        print("Training model took %.2f seconds " % (t1 - t0))

        # compute accuracy on test set
        test_loss, y_hat, params = forward_pass(self.test_data, self.test_labels, params)
        accuracy = compute_accuracy(y_hat, self.test_labels)
        # *** END CODE HERE ***

        self.assertGreater(accuracy, 0.94)

if __name__ == "__main__":

    testSuite = unittest.TestSuite()
    testSuite.addTest(Test_Three_Layer_MLP('test_three_layer_mlp'))

    runner = unittest.TextTestRunner()
    runner.run(testSuite)