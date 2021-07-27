##################################################
## Deep learning crash course, assignment 2
##################################################
## Description : the pytorch autograd
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
import argparse
import unittest
import torch
import numpy as np

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

def scalar_to_scalar_grad(x,y):
    """Compute the gradient of function f(x,y) = x^3+y^3+x^2+y^2+2xy+2x+2y+1

    Inputs:
        x, y: input two scalars
        
    Outputs:
        (dx, dy) : derivatives of f to x and y
    """

    # *** START CODE HERE ***
    vx = torch.tensor(x, requires_grad=True)
    vy = torch.tensor(y, requires_grad=True)
    
    f = vx.pow(3) + vy.pow(3) + vx.pow(2) + vy.pow(2) + 2*vx*vy + 2*vx + 2*vy + 1
    
    f.backward()
    
    return (vx.grad.item(), vy.grad.item())
    # *** END CODE HERE ***
   
def scalar_to_vector_grad(x, A, b):
    """Compute the gradient of function f(x) = 0.5 * x.T * A * x + b*x + 1

    Inputs:
        x: [N, 1] vector
        A: [N, N] weight matrix
        b: [N, 1] linear vector
        
    Outputs:
        (dA, db) : derivatives of f to A and b
    """
    
    # *** START CODE HERE ***
    N = x.shape[0]
    
    assert A.shape[0] == N and A.shape[1] == N and b.shape[0] == N

    vx = torch.tensor(x, requires_grad=False)
    vA = torch.tensor(A, requires_grad=True)
    vb = torch.tensor(b, requires_grad=True)
    
    f = 0.5 * torch.matmul(torch.matmul(vx.T, vA), vx) + torch.matmul(vb.T, vx) + 1
    
    f.backward()
    
    return (vA.grad.numpy(), vb.grad.numpy())
    # *** END CODE HERE ***
    
def tensor_to_tensor_grad(x, A, B, df):
    """Compute the gradient of function f(x) = A*x*B + ReLU(x) to x

    Inputs:
        x: [M, N] matrix
        A: [M, M] weight matrix
        B: [N, N] weight matrix
        df: [M, N], downstream gradient to f

    Outputs:
        dx : derivatives of loss to x, which is the "Jacobian products x"
    """
    
    assert A.shape[0] == x.shape[0] and B.shape[0] == x.shape[1] and df.shape[0]==x.shape[0] and df.shape[1]==x.shape[1]

    # *** START CODE HERE ***
    vx = torch.tensor(x, requires_grad=True)
    vA = torch.tensor(A, requires_grad=False)
    vB = torch.tensor(B, requires_grad=False)
    vdf = torch.tensor(df, requires_grad=False)
    
    f = torch.mm(torch.mm(vA, vx), vB) + torch.nn.functional.relu(vx)
    f.backward(vdf)
    
    return vx.grad.numpy()
    # *** END CODE HERE ***

# ---------------------------------------------

def test_scalar_to_scalar_grad():
    """Test scalar_to_scalar_grad"""
    x = 10.2
    y = 5.6
    
    dx_ground_truth = 3*pow(x,2)+2*x+2*y+2
    dy_ground_truth = 3*pow(y,2)+2*y+2*x+2
    print(f"dx_ground_truth = %f, dy_ground_truth = %f" % (dx_ground_truth, dy_ground_truth))
    
    dx, dy = scalar_to_scalar_grad(x, y)
    print(f"dx = %f, dy = %f" % (dx, dy))
    
    assert abs(dx-dx_ground_truth)<0.1
    assert abs(dy-dy_ground_truth)<0.1
    
def test_scalar_to_vector_grad():
    """Test scalar_to_vector_grad"""

    x = np.random.rand(3,1)
    A = np.random.rand(3,3)
    b = np.random.rand(3,1)
    
    dA, db = scalar_to_vector_grad(x, A, b)
    
    assert np.linalg.norm(dA - 0.5*np.dot(x, x.T))<0.1
    assert np.linalg.norm(db - x)<0.1

def test_tensor_to_tensor_grad():
    """Test tensor_to_tensor_grad"""

    x = np.eye(4,3)-0.5
    A = np.eye(4,4)*2.0
    B = np.eye(3,3)*3.0
    df = np.eye(4,3)-0.24
    
    dx = tensor_to_tensor_grad(x, A, B, df)
       
    dx_ground_truth = np.array([[ 5.32, -1.44, -1.44],
                                [-1.44,  5.32, -1.44],
                                [-1.44, -1.44,  5.32],
                                [-1.44, -1.44, -1.44]])
    
    assert np.linalg.norm(dx - dx_ground_truth)<0.1
    
def main():
    pass

if __name__ == '__main__':
    main()
