import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import numpy as np
from src.models.eql_model import EQLModel, ConnectivityEQLModel
from src.models.custom_functions import SafeIdentityFunction, SafeLog, SafeExp, SafeSin, SafePower
from torch.utils.data import TensorDataset, DataLoader

def generate_data(num_samples=1500):
    """Generate synthetic data for N12."""
    # Create synthetic data with random points in the interval
    x1_values = np.linspace(0, 1, num_samples)
    x2_values = np.linspace(0, 1, num_samples)
    y_values = x1_values**4 + x1_values**3 + 0.5*x2_values**2 - x2_values  # Example function: y = x^2
    return x1_values, x2_values, y_values



def main():
    
    # Define the hypothesis set of unary functions
    hyp_set = [
        SafeIdentityFunction(),  # Identity function
        torch.sin,           # Sine function
        torch.cos,           # Cosine function
        SafeLog(),
        SafeExp(),
        SafeSin(),
        SafePower()
        #torch.sigmoid        # Sigmoid function
    ]

    # Model configuration
    input_size = 2
    output_size = 1
    #hidden_dim = [[2], []] # it is the output size of each neurons in each layer
    num_layers = 2 # hidden + 1 output
    nonlinear_info = [ # it is the number of neurons in each layer
        (4, 0),  # Layer 1: 4 unary, 4 binary functions
        (0, 0),  # Layer 2
        (0, 0)   # Layer 3
    ]

    x1_values, x2_values, y_values = generate_data()

    # Convert to PyTorch tensors
    X1 = torch.tensor(x1_values, dtype=torch.float32).reshape(-1, 1)
    X2 = torch.tensor(x2_values, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(y_values, dtype=torch.float32).reshape(-1, 1)

    # Split data into train and validation sets (80-20 split) using random indices
    indices = np.random.permutation(len(X1))
    train_size = int(0.66666 * len(X1)) + 1
    print(train_size)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    X1_train, X1_val = X1[train_indices], X1[val_indices]
    X2_train, X2_val = X2[train_indices], X2[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    # Create train and validation datasets
    train_dataset = TensorDataset(X1_train, X2_train, y_train)
    val_dataset = TensorDataset(X1_val, X2_val, y_val)

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model with full connectivity, this will be use to search for the best architecture
    model = ConnectivityEQLModel(
        input_size=2,
        output_size=1,
        num_layers=num_layers,
        hyp_set=hyp_set,
        nonlinear_info=nonlinear_info,
        min_connections_per_neuron=1,
        exp_n = 12
    )

    #model.get_equation()

    # Train with parameter optimization
    best_model, best_loss, best_architecture, opt_result = model.train_all_architectures(
        train_loader,
        val_loader,
        num_epochs= 1500,
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

    # Print optimization results
    print(f"Final loss: {opt_result.fun}")
    print(f"Success: {opt_result.success}")
    print(f"Number of iterations: {opt_result.nit}")

    # Print the best architecture
    print(best_model)

    #print(model)
    equation = best_model.get_equation()

    # Evaluate and plot results
    model.eval()
    with torch.no_grad():
        predictions = model(X)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='True Function (sin(x))')
    plt.plot(x_values, predictions.numpy(), '--', label='EQL Prediction')
    plt.legend()
    plt.title('EQL Function Learning Results')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

    # Print learned equation
    equation = model.get_equation()

if __name__ == "__main__":
    main()

