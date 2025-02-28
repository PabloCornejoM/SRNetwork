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
from src.training.connectivity_trainer import ConnectivityTrainer
from src.utils.plotting import plot_results 
from src.utils.data_utils import get_nguyen_data_loaders, generate_nguyen_data
from src.models.model_initialization import initialize_model

def main():
    # Set random seeds for reproducibility
    #torch.manual_seed(42)
    #np.random.seed(42)
    
    # Define the set of functions
    function_set = {
            "identity": SafeIdentityFunction,
            "exp": SafeExp,
            "log": SafeLog,
            "sin": SafeSin,
            "power": SafePower(),
            # Idea: Add "x" function just to know x in the layer
        }

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
    model = initialize_model(
        input_size, 
        output_size, 
        num_layers, 
        function_set, 
        nonlinear_info, 
        min_connections_per_neuron=1, 
        #exp_n=1
    )

    # Training configuration
    config = {
        'training': {
            'num_epochs': 1500,
            'learning_rate': 0.01,
            'reg_strength': 1,
            'decimal_penalty': 0.01,
            'scheduler': 'progressive',  # One of: cosine, cyclic, progressive
            # Connectivity training specific parameters
            'use_connectivity_training': False,  # Set to False for classical training
            'max_architectures': 10,
            'max_patterns_per_layer': 5,
            'num_parallel_trials': 1
        }
    }

    # Train the model using original trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if config['training'].get('use_connectivity_training', False):
        trainer = ConnectivityTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        trained_model, _, _, _ = trainer.train_all_architectures(
            max_architectures=config['training'].get('max_architectures'),
            max_patterns_per_layer=config['training'].get('max_patterns_per_layer'),
            num_parallel_trials=config['training'].get('num_parallel_trials', 3)
        )
    else:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        trainer.train()
        trained_model = model

    # Get the final equation and evaluate results
    equation = trained_model.get_equation()
    print(f"Final equation: {equation}")

    # Evaluate model
    trained_model.eval()
    with torch.no_grad():
        predictions = trained_model(X.to(device))
        predictions = predictions.cpu()

    plot_results(X, y, predictions)

if __name__ == "__main__":
    main() 