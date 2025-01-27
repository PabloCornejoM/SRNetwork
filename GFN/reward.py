import matplotlib.pyplot as pp
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim

import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import numpy as np
from models import EQLModel, ConnectivityEQLModel
from custom_functions import SafeIdentityFunction, SafeLog, SafeExp, SafeSin, SafePower
from torch.utils.data import TensorDataset, DataLoader




def mse_to_score(mse, epsilon=10**-6):
    return np.log(1/(mse + epsilon))

def function_reward(functions, train_loader, val_loader):

    hyp_set = [
        SafeIdentityFunction(),  # Identity function
        torch.sin,               # Sine function
        torch.cos,               # Cosine function
        SafeLog(),
        SafeExp(),
        SafeSin(),
        SafePower()
        #torch.sigmoid        # Sigmoid function
    ]

    # Model configuration
    num_layers = 4 # hidden + 1 output
    nonlinear_info = [ # it is the number of neurons in each layer
        (1, 0),  # Layer 1: 4 unary, 4 binary functions
        (1, 0),  # Layer 2
        (1, 0)   # Layer 3
    ]

    model = ConnectivityEQLModel(
        input_size=1,
        output_size=1,
        num_layers=num_layers,
        hyp_set=hyp_set,
        nonlinear_info=nonlinear_info,
        min_connections_per_neuron=1,
        exp_n = 99,
        functions=functions
    )

    best_model, best_loss, best_architecture, opt_result = model.train_all_architectures(
        train_loader,
        val_loader,
        num_epochs=100,
        max_architectures=10,
        optimize_final=True,  # Enable parameter optimization
        optimization_method='Powell',
        optimization_options={
            'maxiter': 1000,
            'disp': True,
            'adaptive': True,
            'xatol': 1e-8,
            'fatol': 1e-8
        }
    )
    
    return best_loss, mse_to_score(best_loss), best_model