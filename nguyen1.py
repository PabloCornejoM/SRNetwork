import torch
import numpy as np
from models import EQLModel, ConnectivityEQLModel
from custom_functions import SafeIdentityFunction, SafeLog, SafeExp, SafeSin, SafePower
from torch.utils.data import TensorDataset, DataLoader


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
input_size = 1
output_size = 1
hidden_dim = [[2], []] # it is the output size of each neurons in each layer
num_layers = 2 # hidden + 1 output
nonlinear_info = [ # it is the number of neurons in each layer
    (3, 0),  # Layer 1: 4 unary, 4 binary functions
    (0, 0),  # Layer 2
    (0, 0)   # Layer 3
]

# Create synthetic data
x_values = np.linspace(-10, 10, 1000)
y_values = x_values + x_values**2 + x_values**3 # Example function: y = x^2

# Convert to PyTorch tensors
X = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1)
y = torch.tensor(y_values, dtype=torch.float32).reshape(-1, 1)

# Create data loader
dataset = TensorDataset(X, y)
batch_size = 1000
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = ConnectivityEQLModel(
    input_size=1,
    output_size=1,
    num_layers=num_layers,
    hyp_set=hyp_set,
    nonlinear_info=nonlinear_info,
    min_connections_per_neuron=1
)

model.get_equation()

# Train with parameter optimization
best_model, best_loss, best_architecture, opt_result = model.train_all_architectures(
    train_loader,
    num_epochs=100,
    max_architectures=10,
    optimize_final=True,  # Enable parameter optimization
    optimization_method='BFGS',
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
