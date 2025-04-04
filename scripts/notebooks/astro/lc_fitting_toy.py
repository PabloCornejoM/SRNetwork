import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# Add project root to system path - works in both notebook and script
try:
    # For Python scripts
    project_root = str(Path(__file__).parent.parent.parent)
except NameError:
    # For Jupyter notebooks
    project_root = str(Path.cwd().parent.parent)

if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary modules
from src.models.custom_functions import (
    SafeIdentityFunction, SafeLog, SafeExp, 
    SafeSin, SafePower, SafeCos
)
from src.training.trainer import Trainer
from src.training.connectivity_trainer import ConnectivityTrainer
from src.utils.plotting import plot_results
from src.utils.data_utils import get_astro_data_loaders, generate_astro_data
from src.models.model_initialization import initialize_model

# =============================================================================
# Data Loading and Processing Functions
# =============================================================================

def load_equation_info(equation_name, path_to_data='/Users/pablocornejo/Documents/Tesis/SRNetwork/data/raw/astro_sim.txt'):
    """Load equation information from astro_sim.txt file."""
    df = pd.read_csv(path_to_data, sep='\t')
    equation_info = df[df['Equation_name'] == equation_name].iloc[0]
    return equation_info

def add_noise(y, noise_level=0.1):
    """
    Add Gaussian noise to the data.
    
    Args:
        y: Target values
        noise_level: Standard deviation of the noise (default: 0.1)
    
    Returns:
        Noisy target values
    """
    noise = np.random.normal(0, noise_level, y.shape)
    return y + noise

def apply_sparsity(x, y, sparsity_level=0.3):
    """
    Apply sparsity to the data by randomly removing points.
    
    Args:
        x: Input values
        y: Target values
        sparsity_level: Fraction of points to remove (default: 0.3)
    
    Returns:
        Sparse input and target values
    """
    n_points = len(x)
    n_keep = int(n_points * (1 - sparsity_level))
    indices = np.random.choice(n_points, n_keep, replace=False)
    return x[indices], y[indices]

def combine_light_curves(x, y1, y2, weights=(0.7, 0.3)):
    """
    Combine two light curves with specified weights.
    
    Args:
        x: Input values
        y1, y2: Two light curves to combine
        weights: Tuple of weights for each curve (default: (0.7, 0.3))
    
    Returns:
        Combined light curve
    """
    return weights[0] * y1 + weights[1] * y2

# =============================================================================
# Model Configuration and Training
# =============================================================================

def get_model_config():
    """Get the model configuration."""
    return {
        'training': {
            'num_epochs': 1500,
            'learning_rate': 0.01,
            'reg_strength': 0.0001,
            'decimal_penalty': 0.01,
            'scheduler': 'progressive',
            'use_connectivity_training': False,
            'max_architectures': 10,
            'max_patterns_per_layer': 5,
            'num_parallel_trials': 1,
            'print_training_stats': True
        }
    }

def get_model_architecture():
    """Get the model architecture configuration."""
    return {
        'input_size': 1,
        'output_size': 1,
        'num_layers': 2,
        'nonlinear_info': [(2, 0), (0, 0), (0, 0)],  # Two neurons in first layer
        'function_set': {
            "identity": SafeIdentityFunction(),
            "exp": SafeExp(),
            "log": SafeLog(),
            "sin": SafeSin(),
            "cos": SafeCos(),
            "power": SafePower(),
        }
    }

# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_and_fit_light_curve(equation_name, noise_level=0.1, sparsity_level=0.3):
    """
    Process and fit a light curve with noise and sparsity.
    
    Args:
        equation_name: Name of the equation in astro_sim.txt
        noise_level: Level of noise to add
        sparsity_level: Level of sparsity to apply
    """
    # 1. Load and generate data
    print(f"Loading data for equation: {equation_name}")
    X, y = generate_astro_data(equation_name)
    
    # 2. Apply transformations
    print("Applying data transformations...")
    y_noisy = add_noise(y, noise_level)
    X_sparse, y_sparse = apply_sparsity(X, y_noisy, sparsity_level)
    
    # 3. Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = get_data_loaders(X_sparse, y_sparse, batch_size=64)
    
    # 4. Initialize model
    print("Initializing model...")
    model_config = get_model_config()
    arch_config = get_model_architecture()
    
    model = initialize_model(
        arch_config['input_size'],
        arch_config['output_size'],
        arch_config['num_layers'],
        arch_config['function_set'],
        arch_config['nonlinear_info'],
        min_connections_per_neuron=1,
        exp_n=1000
    )
    
    # 5. Train model
    print("Training model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=model_config,
        device=device
    )
    trainer.train()
    
    # 6. Evaluate and plot results
    print("Evaluating results...")
    model.eval()
    with torch.no_grad():
        predictions = model(X.to(device))
        predictions = predictions.cpu()
    
    # Plot original vs noisy vs fitted
    plt.figure(figsize=(15, 5))
    
    # Original data
    plt.subplot(131)
    plt.plot(X.numpy(), y.numpy(), 'b-', label='Original')
    plt.title('Original Data')
    plt.legend()
    
    # Noisy and sparse data
    plt.subplot(132)
    plt.plot(X_sparse.numpy(), y_sparse.numpy(), 'r.', label='Noisy & Sparse')
    plt.title('Noisy & Sparse Data')
    plt.legend()
    
    # Fitted curve
    plt.subplot(133)
    plt.plot(X.numpy(), y.numpy(), 'b-', label='Original')
    plt.plot(X.numpy(), predictions.numpy(), 'r--', label='Fitted')
    plt.title('Fitted Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print final equation
    equation = model.get_equation()
    print(f"\nFinal equation: {equation}")
    
    return model, equation

# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Process and fit toy-1 equation
    model, equation = process_and_fit_light_curve(
        equation_name='toy-1',
        noise_level=0.1,
        sparsity_level=0.3
    )
