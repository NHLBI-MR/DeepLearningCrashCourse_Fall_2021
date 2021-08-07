##################################################
## Deep learning crash course, assignment 2
##################################################
## Description : the N-layer MLP model, using pytorch
## Author: Hui Xue
## Copyright: 2021, All rights reserved
## Version: 1.0.1
## Maintainer: xueh2@github
## Email: hui.xue@nih.gov
## Status: active development
##################################################

import wandb

# set up the sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {
      'name': 'accu_test',
      'goal': 'maximize'   
    },
    'parameters': {
        'seeds': {
            'min':1,
            'max':2**32
        },
        'epochs': {
            'values': [30, 40, 50, 60, 70]
        },
        'batch_size': {
            'values': [256, 512]
        },
        'learning_rate': {
            'min':5e-2,
            'max':2e-1
        },
        'reg': {
            'values': [0.0, 1e-3, 0.0025, 0.005, 1e-2, 5e-2, 1e-1]
        },
        'num_hidden_layers':{
            'values':[[300, 200, 100], [300, 300, 200, 100], [200, 100], [300, 300], [200, 200, 200, 200, 100]]
        },
        'optimizer': {
            'values': ['adam', 'sgd']
        },
        'scheduler': {
            'values': ['one_cycle', 'step']
        },
    }
}

def main():

    sweep_id = wandb.sweep(sweep_config, project="A2_Pytorch_MLP_Sweep")
    
if __name__ == '__main__':
    main()