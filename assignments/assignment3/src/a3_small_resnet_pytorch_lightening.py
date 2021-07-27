##################################################
## Deep learning crash course, assignment 3
##################################################
## Description : a small resnet model, using pytorch lightening and wandb
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Mmaintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse
from time import gmtime, strftime

from pytorch_lightning import callbacks

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

from cifar10dataset import *
import util
import resnet_model
import train

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, sampler
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

# ----------------------------------
def add_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pytorch CNN model for Cifar10 classification, using wandb")

    parser.add_argument('--min_epochs', type=int, default=10, help='minimal number of epochs to train')
    parser.add_argument('--max_epochs', type=int, default=30, help='maximal number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--reg', type=float, default=0.005, help='regularization lambda')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='learn rate')
    parser.add_argument('--gpus', type=int, default=2, help='set gpus to use; 0 is to use cpu')

    parser.add_argument(
        "--training_record",
        type=str,
        default="pytorch_small_resnet",
        help='String to record this training')

    parser.add_argument(
        "--accelerator",
        type=str,
        default="dp",
        help='Distribution method')

    return parser
# ----------------------------------

num_samples_validation = 3000

# ----------------------------------

# load parameters
args = add_args().parse_args()
print(args)

# ----------------------------------
# data module

class Cifar10DataModule(pl.LightningDataModule):

    def __init__(self, batch_size=256):
        super().__init__()
        self.cifar10_dataset = util.load_and_prepare_data(os.path.join(Project_DIR, "../data/cifar10"), subtract_mean=True)
        self.batch_size = batch_size

    def setup(self, stage = None):
        self.train_set, self.test_set, self.loader_for_train, self.loader_for_val, self.loader_for_test = set_up_cifar_10_dataset(self.cifar10_dataset, num_samples_validation=num_samples_validation, batch_size=self.batch_size)

        self.loader_for_train.num_workers = os.cpu_count()
        self.loader_for_val.num_workers = os.cpu_count()
        self.loader_for_test.num_workers = os.cpu_count()

    def train_dataloader(self):
        return self.loader_for_train

    def val_dataloader(self):
        return self.loader_for_val

    def test_dataloader(self):
        return self.loader_for_test

# ----------------------------------
# implement the pytorch lighetening module
# you can still use the resnet_model.ResBlock and resnet_model.ResBlockDownSample

class Cifar10SmallResNetLightning(pl.LightningModule):

    def __init__(self, H=32, W=32, C=3, batch_size=2048, learning_rate=0.002, reg=0.005):
        """Initial the model

        Create the same network as the Cifar10SmallResNet

        Args:
            H (int): Height of input image
            W (int): Width of input image
            C (int): Number of channels of input image
        """
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = True

        # *** START CODE HERE ***

        self.input_conv = nn.Conv2d(C, 32, kernel_size=(5, 5), stride=1, padding=(2, 2), bias=True)

        self.b1 = resnet_model.ResBlock(3, 3, 32, 32, 'block_1')
        self.b2 = resnet_model.ResBlock(3, 3, 32, 32, 'block_2')
        self.b3 = resnet_model.ResBlockDownSample(3, 3, 32, 64, 'blockdownsample_1')

        self.b4 = resnet_model.ResBlock(3, 3, 64, 64, 'block_3')
        self.b5 = resnet_model.ResBlock(3, 3, 64, 64, 'block_4')
        self.b6 = resnet_model.ResBlockDownSample(3, 3, 64, 128, 'blockdownsample_2')

        self.b7 = resnet_model.ResBlock(3, 3, 128, 128, 'block_5')
        self.b8 = resnet_model.ResBlock(3, 3, 128, 128, 'block_6')
        self.b9 = resnet_model.ResBlockDownSample(3, 3, 128, 256, 'blockdownsample_3')

        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.relu1 = nn.ReLU()
        self.b10 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=True)

        self.bn2 = nn.BatchNorm2d(num_features=512)
        self.relu2 = nn.ReLU()
        self.b11 = nn.Conv2d(512, 512, kernel_size=(4, 4), stride=1, padding=(0, 0), bias=True)

        input_dim = 512

        self.fc256 = nn.Linear(input_dim, 256, bias=True)
        self.bn3 = nn.BatchNorm1d(num_features=256)
        self.relu3 = nn.ReLU()
        self.output_fc10 = nn.Linear(256, 10, bias=True)

        # initialize the conv layer weights
        nn.init.kaiming_normal_(self.input_conv.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.b10.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.b11.weight, mode='fan_out')

        # initialize the fc layer weights
        nn.init.xavier_normal_(self.fc256.weight)
        nn.init.xavier_normal_(self.output_fc10.weight)

        # create loss
        self.loss_func = nn.CrossEntropyLoss()
        # *** END CODE HERE ***

    def forward(self, x):
        """Forward pass

        Args:
            x ([B, C, H, W]): a batch of input image

        Returns:
            output ([B, 10]): logits tensor, ready for the softmax
        """
        # *** START CODE HERE ***
        x = self.input_conv(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        x = self.b7(x)
        x = self.b8(x)
        x = self.b9(x)

        x = self.bn1(x)
        x = self.relu1(x)
        x = self.b10(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.b11(x)

        x = self.fc256(torch.flatten(x, 1))
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.output_fc10(x)

        return x
        # *** END CODE HERE ***

    def training_step(self, batch, batch_idx):
        # *** START CODE HERE ***
        x, y = batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        # *** END CODE HERE ***
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # *** START CODE HERE ***
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_func(y_hat, y)
        val_acc = util.compute_accuracy(y_hat, y)
        # *** END CODE HERE ***
        self.log('validation_loss', val_loss, prog_bar=True)
        self.log('validation_acc', val_acc, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        # *** START CODE HERE ***
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss_func(y_hat, y)
        test_acc = util.compute_accuracy(y_hat, y)
        # *** END CODE HERE ***
        self.log('test_loss', test_loss, prog_bar=True)
        self.log('test_acc', test_acc, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        # *** START CODE HERE ***
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=self.hparams.reg)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.5, last_epoch=-1, verbose=False)
        # *** END CODE HERE ***
        return {'optimizer': optimizer, 
                'lr_scheduler': {
                    'scheduler': scheduler,
                    "interval": "epoch", 
                    "frequency": 1
                    }
               }

# ----------------------------------

def run_training():
    """Run the training

    Outputs:
        model : best model after training
        loss_train, loss_val : loss for every epoch
        accu_train, accu_val : accuracy for every epoch
    """

    # Initialize a new wandb run
    wandb_logger = WandbLogger(name='pytorch_lightening_small_resnet') 

    # call pytorch lightening training
    data_module = Cifar10DataModule(batch_size=args.batch_size)
    model = Cifar10SmallResNetLightning(H=32, W=32, C=3, batch_size=args.batch_size, learning_rate=args.learning_rate, reg=args.reg)
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.max_epochs, min_epochs=args.min_epochs, logger=wandb_logger, auto_scale_batch_size=False, accelerator=args.accelerator)

    # perform training
    trainer.fit(model, datamodule=data_module)
    # do the test
    res_testing = trainer.test(datamodule=data_module)

    return model, res_testing[0]['test_loss'], res_testing[0]['test_acc']

def main():

    # perform training
    m, loss_test, accu_test = run_training()

    # print out accuracy
    print('Test accuracies are %f for experiment run %s' % (accu_test, args.training_record))
    
if __name__ == '__main__':
    main()