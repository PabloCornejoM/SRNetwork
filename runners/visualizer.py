import torch
import numpy as np
import sys
from pathlib import Path

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


from models import EQLModel, ConnectivityEQLModel
from custom_functions import SafeIdentityFunction, SafeLog, SafeExp, SafeSin, SafePower
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

#from losscape.train import train
from losscape.create_landscape import create_2D_losscape

device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_data(num_samples=15000):
    """Generate synthetic data for x^2."""
    x_values = np.linspace(-10, 10, num_samples)
    y_values = x_values + x_values**2 + x_values**3
    return x_values, y_values

def create_2D_losscape2(model, train_loader, output_vtp=True):
    """Create and visualize 2D loss landscape."""
    # Get reference point (current model parameters)
    model.eval()
    
    # Generate evaluation points
    x_values, y_values = generate_data()
    X = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(y_values, dtype=torch.float32).reshape(-1, 1)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='True Function')
    
    with torch.no_grad():
        predictions = model(X)
        plt.plot(x_values, predictions.numpy(), '--', label='Model Prediction')
    
    plt.legend()
    plt.title('Model Prediction vs True Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    
    if output_vtp:
        plt.savefig('loss_landscape.png')
    plt.show()
    
    # Print learned equation
    print("\nLearned Equation:")
    print(model.get_equation())

def train(model, train_loader, optimizer:torch.optim = None, criterion = F.cross_entropy, epochs:int = 50, decay_lr_epochs:int = 20, verbose:int = 1):
    """
    Train the provided model.

    Parameters
    ----------
    model : the torch model which will be trained.
    train_loader : the torch dataloader which gives training data.
    optimizer : the optimizer used for training (should follow the same API as torch optimizers).(default to Adam)
    criterion : the criterion used to compute the loss. (default to F.cross_entropy)
    epochs : the number of epochs (default to 50)
    decay_lr_epochs : the lr will be divided by 10 every decay_lr_epochs epochs (default to 20)
    verbose : controls the printing during the training. (0 = print at the end only, 1 = print at 0,25,50,100%, 2 = print every epoch). (default to 1)

    Returns
    ----------

    """

    model.to(device)

    if criterion is None:
        criterion = F.cross_entropy
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []  # Track losses
    model.train()  # Explicitly set model to training mode
    
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (Xb, Yb) in enumerate(train_loader):
            Xb, Yb = Xb.to(device), Yb.to(device)

            logits = model(Xb)
            # Change criterion to MSE since this is a regression problem
            loss = F.mse_loss(logits, Yb) if criterion == F.cross_entropy else criterion(logits, Yb)
            
            # Check if loss is valid
            if not torch.isfinite(loss):
                print(f"Warning: Invalid loss value detected: {loss.item()}")
                continue
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        losses.append(avg_epoch_loss)

        if epoch%decay_lr_epochs == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        
        """
        if int(epoch) == 150 or int(epoch) == 225 or int(epoch) == 275:
            lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
        """


        if verbose == 2:
            print(f"Epoch {epoch}/{epochs}. Loss={avg_epoch_loss:.8f}")

        if verbose == 1 and (epoch%(epochs/4) == 0):
            print(f"Epoch {epoch}/{epochs}. Loss={avg_epoch_loss:.8f}")

    if verbose == 0:
        print(f"Final Epoch {epoch}/{epochs}. Loss={avg_epoch_loss:.8f}")
        
    return losses  # Return loss history

def setup_default_model():
    """Create a default model configuration."""
    hyp_set = [
        SafeIdentityFunction(),
        torch.sin,
        torch.cos,
        SafeLog(),
        SafeExp(),
        SafeSin(),
        SafePower()
    ]
    
    nonlinear_info = [
        (3, 0),  # Layer 1: 3 unary, 0 binary functions
        (0, 0),  # Layer 2
        (0, 0)   # Layer 3
    ]
    
    return ConnectivityEQLModel(
        input_size=1,
        output_size=1,
        num_layers=2,
        hyp_set=hyp_set,
        nonlinear_info=nonlinear_info,
        min_connections_per_neuron=1,
        exp_n=1
    )

if __name__ == "__main__":
    # Example usage
    model = setup_default_model()
    x_values, y_values = generate_data()
    X = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(y_values, dtype=torch.float32).reshape(-1, 1)
    
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    
    train(model, train_loader, epochs = 1000, verbose = 1) # losscape can perform the training for you
    create_2D_losscape(model, train_loader, output_vtp=True)
