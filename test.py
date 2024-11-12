import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from models_pytorch import EQLModel

# Define the hypothesis set of unary functions
hyp_set = [
    torch.nn.Identity(),  # Identity function
    torch.sin,           # Sine function
    torch.cos,           # Cosine function
    #torch.sigmoid        # Sigmoid function
]

# Model configuration
input_size = 1
output_size = 1
num_layers = 4
nonlinear_info = [
    (2, 0),  # Layer 1: 4 unary, 4 binary functions
    (40, 0),  # Layer 2
    (3, 0)   # Layer 3
]

# Create synthetic data
x_values = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y_values = np.sin(x_values)  # Example function: y = sin(x)

# Convert to PyTorch tensors
X = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1)
y = torch.tensor(y_values, dtype=torch.float32).reshape(-1, 1)

# Create data loader
dataset = TensorDataset(X, y)
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model
model = EQLModel(
    input_size=input_size,
    output_size=output_size,
    num_layers=num_layers,
    hyp_set=hyp_set,
    nonlinear_info=nonlinear_info
)

print(model)

# Training parameters
num_epochs = 170  # Total epochs (matches the three phases: 25% + 70% + 5%)
learning_rate = 0.001
reg_strength = 1e-3
threshold = 0.1

# Train the model
from models_pytorch import train_eql_model
train_eql_model(
    model=model,
    train_loader=train_loader,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    reg_strength=reg_strength,
    threshold=threshold
)

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
