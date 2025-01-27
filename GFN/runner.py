import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.categorical import Categorical
import tqdm
from gflownet import FlowModel, function_to_tensor, face_parents, sorted_keys
from reward import function_reward
from train_gflow import train_gflow, sample_gflow
import matplotlib.pyplot as plt



def generate_data(num_samples=1500):
    """Generate synthetic data for desired function."""
    # Create synthetic data with random points in the interval
    x_values = np.linspace(0.00001, 2, num_samples)
    y_values = np.log(np.sin(x_values))
    return x_values, y_values





def main():
    x_values, y_values = generate_data()

    X = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(y_values, dtype=torch.float32).reshape(-1, 1)

    # Split data into train and validation sets (80-20 split) using random indices
    indices = np.random.permutation(len(X))
    train_size = int(0.66666 * len(X)) + 1
    print(train_size)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    # Create train and validation datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    ######################################################################################

    F_sa = FlowModel(512)
    opt = torch.optim.Adam(F_sa.parameters(), 3e-4)

    # Call the training function
    losses, sampled_faces = train_gflow(F_sa, train_loader, val_loader, opt)

    plt.figure(figsize=(10,3))
    plt.plot(losses)
    plt.yscale('log')
    plt.show()

    sampled_faces = sample_gflow(F_sa, train_loader, val_loader)

if __name__ == "__main__":
    main()