import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models import ConnectivityEQLModel
from custom_functions import SafeIdentityFunction, SafeSin, SafePower, SafeLog, SafeExp
from utils.wandb_logger import WandBLogger

def generate_data(num_samples=1000):
    """Generate synthetic data for x^2."""
    x = np.random.uniform(-1, 1, (num_samples, 1))
    y = (x + x**2 + x**3)
    return x, y

def main():
    # Set random seed for reproducibility
    #torch.manual_seed(42)
    #np.random.seed(42)
    
    # Generate data
    x_data, y_data = generate_data()
    
    # Convert to PyTorch tensors
    x_tensor = torch.FloatTensor(x_data)
    y_tensor = torch.FloatTensor(y_data)
    
    # Create dataset and dataloader
    batch_size = 64
    dataset = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Define the hypothesis set (available functions)
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
    
    # Create model
    model = ConnectivityEQLModel(
        input_size=1,
        output_size=1,
        num_layers=2,  # hidden + 1 output
        hyp_set=hyp_set,
        nonlinear_info=[ # it is the number of neurons in each layer
            (3, 0),  # Layer 1: 4 unary, 4 binary functions
            (0, 0),  # Layer 2
            (0, 0)   # Layer 3
        ],  # One unary node, no binary nodes
        name='sin_approximation',
        min_connections_per_neuron=1,
    )
    
    # Configure wandb logger
    config = {
        "learning_rate":  0.001,
        "num_epochs": 1500,
        "batch_size": batch_size,
        "model_architecture": str(model),
        "regularization_strength": 1e-3,
        "threshold": 0.1,
        "dataset_size": len(x_data),
        "input_range": "[-1, 1]",
        "target_function": "(x^2)",
        "hypothesis_set": [f.__class__.__name__ for f in hyp_set]
    }
    
    logger = WandBLogger(
        project_name="eql-experiments",
        config=config,
        run_name="pow_analysis",
        notes="Discovering pow(x) function using EQL"
    )
    
    # Log model architecture
    logger.log_model_architecture(model)
    
    # Train model
    '''train_eql_model(
        model=model,
        train_loader=train_loader,
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        reg_strength=config["regularization_strength"],
        threshold=config["threshold"],
        logger=logger
    )'''

    # Train with parameter optimization
    best_model, best_loss, best_architecture, opt_result = model.train_all_architectures(
        train_loader,
        val_loader,
        num_epochs=config["num_epochs"],
        max_architectures=10,
        optimize_final=True,  # Enable parameter optimization
        optimization_method='Powell',
        optimization_options={
            'maxiter': 1000,
            'disp': True,
            'adaptive': True,
            'xatol': 1e-8,
            'fatol': 1e-8
        },
        logger=logger  # Pass the logger to the training function
    )

    
    # Generate predictions for visualization
    best_model.eval()
    with torch.no_grad():
        y_pred = best_model(x_tensor).numpy()
    
    # Log final prediction plot
    logger.log_prediction_plot(y_data, y_pred, step=config["num_epochs"])
    
    # Get and log the final equation
    final_equation = best_model.get_equation()
    print("\nFinal discovered equation:")
    print(final_equation)
    logger.log_equation(final_equation, step=config["num_epochs"])
    
    # Finish logging
    logger.finish()

    # Print optimization results
    print(f"Final loss: {opt_result.fun}")
    print(f"Success: {opt_result.success}")
    print(f"Number of iterations: {opt_result.nit}")

if __name__ == "__main__":
    main()