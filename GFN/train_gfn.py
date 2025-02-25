import torch
import argparse

import matplotlib.pyplot as plt
from gflownet.env.sr_env import SRTree
from policy import RNNForwardPolicy, CanonicalBackwardPolicy
from gflownet import GFlowNet, trajectory_balance_loss
from tqdm import tqdm


from functions import Functions
from src.data.nguyen_data import get_nguyen_data_loaders, generate_nguyen_data
from sr_network import SRNet


def main():
    # Initialize the function set
    functions = Functions()

    # Generate data from experiment
    train_loader, val_loader = get_nguyen_data_loaders('Nguyen-1', batch_size=64)
    
    # Get the full dataset for plotting
    X, y = generate_nguyen_data('Nguyen-1')

    # Initialize the environment
    input_size = X.shape[1]
    output_size = y.shape[1]
    num_layers = 2
    nonlinear_info = [(3, 0), (0, 0), (0, 0)] # Network Structure
    env = SRNet()

if __name__ == "__main__":
    main()