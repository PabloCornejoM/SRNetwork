import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

def load_equation_info(equation_name, path_to_data):
    """Load equation information from nguyen.txt file."""
    # Read the tab-separated file
    df = pd.read_csv(path_to_data, sep='\t')
    
    # Get equation info for the specified name
    equation_info = df[df['Equation_name'] == equation_name].iloc[0]
    return equation_info

def generate_input_data(n_points, min_range, max_range, n_inputs, sampling='Uniform'):
    """Generate input data based on specifications."""
    if sampling.lower() == 'uniform':
        x = np.random.uniform(min_range, max_range, size=(n_points, n_inputs))
    else:
        raise ValueError(f"Sampling method {sampling} not implemented")
    return x

def generate_nguyen_data(equation_name, path_to_data=None):
    """Generate synthetic data for any Nguyen equation."""
    if path_to_data is None:
        path_to_data = '/Users/pablocornejo/Documents/Tesis/SRNetwork/data/raw/nguyen.txt'
    # Load equation information
    eq_info = load_equation_info(equation_name, path_to_data)

    # Generate input data
    x = generate_input_data(
        n_points=eq_info['n_points'],
        min_range=eq_info['min_range'],
        max_range=eq_info['max_range'],
        n_inputs=eq_info['in_f'],
        sampling=eq_info['sampling']
    )
    
    # Convert to tensor for computation
    x_tensor = torch.from_numpy(x).float()
    
    # Evaluate the equation using numpy (safer than eval)
    y = eval(eq_info['Equation'])
    
    # Convert to tensors and ensure correct dimensions
    x_tensor = torch.from_numpy(x).float().view(-1, eq_info['in_f'])
    y_tensor = torch.from_numpy(y).float().view(-1, 1)
    
    return x_tensor, y_tensor
    

def generate_astro_data(equation_name, path_to_data=None):
    """Generate synthetic data for any Astro equation."""
    if path_to_data is None:
        path_to_data = '/Users/pablocornejo/Documents/Tesis/SRNetwork/data/raw/astro_sim.txt'
    # Load equation information
    eq_info = load_equation_info(equation_name, path_to_data)
    
    # Generate input data
    x = generate_input_data(
        n_points=eq_info['n_points'],
        min_range=eq_info['min_range'],
        max_range=eq_info['max_range'],
        n_inputs=eq_info['in_f'],
        sampling=eq_info['sampling']
    )
    
    # Convert to tensor for computation
    x_tensor = torch.from_numpy(x).float()
    
    # Evaluate the equation using numpy (safer than eval)
    y = eval(eq_info['Equation'])
    
    # Convert to tensors and ensure correct dimensions
    x_tensor = torch.from_numpy(x).float().view(-1, eq_info['in_f'])
    y_tensor = torch.from_numpy(y).float().view(-1, 1)
    
    return x_tensor, y_tensor

def create_datasets(x_values, y_values, train_ratio=0.66666):
    """Split data into training and validation datasets."""
    indices = np.random.permutation(len(x_values))
    train_size = int(train_ratio * len(x_values)) + 1
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    X_train, X_val = x_values[train_indices], x_values[val_indices]
    y_train, y_val = y_values[train_indices], y_values[val_indices]

    # No need to convert to tensor again as inputs are already tensors
    return TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)

def create_data_loaders(train_dataset, val_dataset, batch_size=64):
    """Create data loaders for training and validation datasets."""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Example usage function
def get_data_loaders(X, y, batch_size=64, train_ratio=0.66666):
    """Convenience function to get data loaders for a specific equation."""
    x_values, y_values = X, y
    train_dataset, val_dataset = create_datasets(x_values, y_values, train_ratio)
    return create_data_loaders(train_dataset, val_dataset, batch_size) 

