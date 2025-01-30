import sys
from pathlib import Path

# Ensure the project root is in the system path
def add_project_root_to_sys_path():
    project_root = str(Path(__file__).parent.parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

add_project_root_to_sys_path()

import torch
import numpy as np
from src.models.custom_functions import SafeIdentityFunction, SafeLog, SafeExp, SafeSin, SafePower
from src.training.trainer import Trainer
from src.utils.plotting import plot_results 
from src.utils.data_utils import get_nguyen_data_loaders, generate_nguyen_data
from src.models.model_initialization import initialize_model  # Import the new initialize_model function



def main():
    # Define the hypothesis set of unary functions
    hyp_set = [
        SafeIdentityFunction(),
        torch.sin,
        torch.cos,
        SafeLog(),
        SafeExp(),
        SafeSin(),
        SafePower()
    ]

    # Model configuration
    input_size = 1  # Nguyen-1 is a single input function
    output_size = 1
    num_layers = 2
    nonlinear_info = [(3, 0), (0, 0), (0, 0)]

    # Get data loaders using the new utility function
    train_loader, val_loader = get_nguyen_data_loaders('Nguyen-1', batch_size=64)
    
    # Get the full dataset for plotting
    X, y = generate_nguyen_data('Nguyen-1')

    # Initialize model
    model = initialize_model(input_size, output_size, num_layers, hyp_set, nonlinear_info, min_connections_per_neuron=1, exp_n=1)


    # Training configuration
    config = {
        'training': {
            'num_epochs': 1500,
            'learning_rate': 0.01,
            'reg_strength': 1e-3,
            'decimal_penalty': 0.01,
            'scheduler_step_size': 500,
            'scheduler_gamma': 0.5
        }
    }

    # Initialize trainer and train the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        config=config, 
        device=device
    )
    trainer.train()

    # Get the final equation and evaluate results
    equation = model.get_equation()
    print(f"Final equation: {equation}")

    # Evaluate model
    model.eval()
    with torch.no_grad():
        predictions = model(X.to(device))
        predictions = predictions.cpu()

    plot_results(X, y, predictions)  # Call the imported function

if __name__ == "__main__":
    main()
