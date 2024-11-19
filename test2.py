import torch
import numpy as np
from models import EQLModel
from custom_functions import SafeLog, SafeExp, SafeSin

def test_simple_function():
    """Test EQL model on a simple function with the new activation functions."""
    
    # Generate synthetic data
    x = np.linspace(-2, 2, 1000).reshape(-1, 1)
    # Example: y = sin(x) + log(|x| + 1) + exp(-x)
    y = np.sin(x) + np.log(np.abs(x) + 1) + np.exp(-x)
    
    # Convert to PyTorch tensors
    x_tensor = torch.FloatTensor(x)
    y_tensor = torch.FloatTensor(y)
    
    # Create model with custom functions
    model = EQLModel(
        input_size=1,
        output_size=1,
        hidden_dim=[3],  # One hidden layer with 3 nodes
        num_layers=2,
        hyp_set=[
            torch.nn.Identity(),
            SafeSin(),
            SafeLog(),
            SafeExp()
        ],
        nonlinear_info=[(3, 1)]  # 3 unary nodes, 1 binary node
    )
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train model
    from models import train_eql_model
    train_eql_model(
        model=model,
        train_loader=train_loader,
        num_epochs=100,
        learning_rate=0.001,
        reg_strength=1e-4,
        threshold=0.1
    )
    
    # Get learned equation
    equation = model.get_equation()
    print("\nLearned equation:")
    print(equation)
    
    return model, equation

def test_multivariate_function():
    """Test EQL model on a multivariate function with the new activation functions."""
    
    # Generate synthetic data for 2D input
    x1 = np.linspace(-2, 2, 30)
    x2 = np.linspace(-2, 2, 30)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.column_stack((X1.flatten(), X2.flatten()))
    
    # Example: y = sin(x1) * log(|x2| + 1) + exp(-x1*x2)
    Y = (np.sin(X[:, 0]) * np.log(np.abs(X[:, 1]) + 1) + 
         np.exp(-X[:, 0] * X[:, 1])).reshape(-1, 1)
    
    # Convert to PyTorch tensors
    x_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(Y)
    
    # Create model with custom functions
    model = EQLModel(
        input_size=2,
        output_size=1,
        hidden_dim=[4, 3],  # Two hidden layers
        num_layers=3,
        hyp_set=[
            torch.nn.Identity(),
            SafeSin(),
            SafeLog(),
            SafeExp()
        ],
        nonlinear_info=[(4, 2), (3, 1)]  # Layer configs
    )
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Train model
    from models import train_eql_model
    train_eql_model(
        model=model,
        train_loader=train_loader,
        num_epochs=200,
        learning_rate=0.001,
        reg_strength=1e-4,
        threshold=0.1
    )
    
    # Get learned equation
    equation = model.get_equation()
    print("\nLearned equation:")
    print(equation)
    
    return model, equation

if __name__ == "__main__":
    print("Testing simple function...")
    model_simple, eq_simple = test_simple_function()
    
    print("\nTesting multivariate function...")
    model_multi, eq_multi = test_multivariate_function()