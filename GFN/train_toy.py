import torch
import argparse

import matplotlib.pyplot as plt
from gflownet import GFlowNet, trajectory_balance_loss
from tqdm import tqdm
import sys
from pathlib import Path

# Ensure the project root is in the system path
def add_project_root_to_sys_path():
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

add_project_root_to_sys_path()

from functions import Functions, get_default_function_categories, get_function_categories_from_names
from src.utils.data_utils import get_nguyen_data_loaders, generate_nguyen_data
from src.models.eql_model import SRNetwork
from policy import RNNForwardPolicy, CanonicalBackwardPolicy
from src.training.trainer import Trainer
from reward import compute_reward
from log import Log

# Global variables for tracking best equation
best_reward_so_far = float('-inf')
best_equation_so_far = None
best_equation_epoch = 0

def train_plot(errs, flows, avg_mses, top_mses):
    """Plot the training curves"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.plot(errs)
    plt.title('Loss')
    plt.xlabel('Iteration')
    
    plt.subplot(132)
    plt.plot(flows)
    plt.title('Total Flow')
    plt.xlabel('Iteration')
    
    plt.subplot(133)
    plt.plot(avg_mses, label='Median MSE')
    plt.plot(top_mses, label='Top 10% MSE')
    plt.title('MSE')
    plt.xlabel('Iteration')
    plt.legend()
    
    plt.tight_layout()
    plt.show()


def main():
    # Initialize the function set
    function_handler = Functions()
    functions = function_handler.functions
    
    # Create function categories based on function names
    function_names = list(functions.keys())
    
    # Define function types
    function_types = {
        'activation': ['sin', 'exp', 'log'],
        'linear': ['identity', 'power'],
        'input_only': ['identity', 'power'],
        'hidden_only': ['sin', 'exp', 'log', 'identity', 'power'],
        'output_only': ['identity', 'sin', 'exp', 'log']
    }
    
    # Get function categories mapping types to indices
    function_categories = get_function_categories_from_names(function_names, function_types)
    
    # Generate data from experiment
    train_loader, val_loader = get_nguyen_data_loaders('Nguyen-1', batch_size=64)
    
    # Get the full dataset for plotting
    X, y = generate_nguyen_data('Nguyen-1')

    # Initialize the environment
    input_size = X.shape[1]
    output_size = y.shape[1]
    num_layers = 2
    nonlinear_info = [(3, 0)]  # Network Structure (3 unary functions, 0 binary functions)
    
    # Initialize the environment
    env = SRNetwork(input_size, output_size, num_layers, functions, nonlinear_info)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 1  # Increased batch size for better statistics

    # Initialize policies
    forward_policy = RNNForwardPolicy(
        batch_size=batch_size, 
        hidden_dim=128, 
        num_functions=len(functions),
        num_layers=1, 
        model="lstm", 
        device=device,
        function_categories=function_categories
    )
    
    config = {
        'training': {
            'num_epochs': 200,
            'learning_rate': 0.01,
            'reg_strength': 1,
            'decimal_penalty': 0.01,
            'scheduler': 'progressive',  # One of: cosine, cyclic, progressive
            # Connectivity training specific parameters
            'use_connectivity_training': False,  # Set to False for classical training
            'max_architectures': 10,
            'max_patterns_per_layer': 5,
            'num_parallel_trials': 1,
            'print_training_stats': False
        }
    }

    backward_policy = CanonicalBackwardPolicy(len(functions))
    
    # Initialize GFlowNet
    model = GFlowNet(forward_policy, backward_policy, env)
    
    # Setup optimizer
    params = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)
    
    # Tracking metrics
    flows, errs, avg_mses, top_mses = [], [], [], []

    num_epochs = 500
    show_plot = True

    for i in (p := tqdm(range(num_epochs))):
        # Initialize starting states
        s0 = env.initialize_functions_structure(batch_size)
        
        # Sample states using GFlowNet
        states, log_probs, action_sequences, total_flow = model.sample_states(s0)
        env.create_structure_model(states)

        rewards = reward(states)

        log = Log(states, log_probs, rewards, total_flow)
        
        # Calculate loss
        loss = trajectory_balance_loss(
            log.total_flow,
            log.rewards,
            log.fwd_probs
        )
        
        # Backpropagation
        loss.backward()
        opt.step()
        opt.zero_grad()

        # Track metrics periodically
        if i % 20 == 0:
            p.set_description(f"Loss: {loss.item():.3f}")
            flows.append(log.total_flow.item())
            errs.append(loss.item())
            avg_mse, top_mse = evaluate_model(
                env=env,
                model=model,
                eval_bs=10,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device,
                current_epoch=i  # Pass current epoch to track when best equation was found
            )
            avg_mses.append(avg_mse.item())
            top_mses.append(top_mse.item())

    # Print final best equation
    if best_equation_so_far is not None:
        print("\nFinal best equation found:")
        print(f"Epoch: {best_equation_epoch}")
        print(f"Reward: {best_reward_so_far:.4f}")
        print(f"Equation:\n{best_equation_so_far}")
        print("-" * 50)

    # Plot training metrics
    if show_plot:
        train_plot(errs, flows, avg_mses, top_mses)

    return model, env, errs, avg_mses, top_mses


def evaluate_model(env, model, eval_bs: int = 20, top_quantile: float = 0.1, train_loader=None, val_loader=None, config=None, device=None, current_epoch=None):
    """
    Evaluate the model by sampling states and calculating rewards based on power function count.
    
    Args:
        env: The environment/model being optimized
        model: The GFlowNet model
        eval_bs: Batch size for evaluation
        top_quantile: Quantile for top reward calculation
        current_epoch: Current training epoch number
        
    Returns:
        avg_reward: Median reward of sampled states
        top_reward: Top percentile reward of sampled states
    """
    global best_reward_so_far, best_equation_so_far, best_equation_epoch
    
    # Initialize states for evaluation
    eval_s0 = env.initialize_functions_structure(eval_bs)
    
    # Sample states
    eval_s, log_probs, action_sequences, total_flow = model.sample_states(eval_s0)
    
    # Calculate rewards for each state using our toy reward function
    rewards = reward(eval_s)
    
    # Filter out any invalid results
    rewards = rewards[torch.isfinite(rewards)]
    
    # Calculate statistics
    avg_reward = torch.median(rewards)
    top_reward = torch.quantile(rewards, q=top_quantile)
    
    # Track best structure
    if current_epoch is not None:
        best_reward_in_batch = torch.max(rewards)
        if best_reward_in_batch > best_reward_so_far:
            best_reward_so_far = best_reward_in_batch
            best_idx = torch.argmax(rewards)
            
            # Create the model with the best state to get its equation
            env.create_structure_model([eval_s[best_idx]])
            best_equation_so_far = env.get_equation()
            best_equation_epoch = current_epoch
            
            print(f"\nNew best structure found at epoch {current_epoch}:")
            print(f"Reward: {best_reward_so_far:.4f}")
            print(f"Equation:\n{best_equation_so_far}")
            print("-" * 50)
    
    return avg_reward, top_reward


def reward(states):
    """
    Compute a toy reward for each state based on function composition.
    Power functions get highest reward, but other functions also contribute.
    
    Args:
        states: List of states, where each state is a list of layers containing function indices
        
    Returns:
        torch.Tensor: Rewards for each state
    """
    batch_size = len(states)
    rewards = torch.zeros(batch_size)
    
    # Function scores dictionary
    # Indices based on the function types defined in main():
    # 0: sin, 1: exp, 2: log, 3: power, 4: identity
    function_scores = {
        0: 0.0,  # sin
        1: 1.5,  # exp
        2: 1.0,  # log
        3: 4.0,  # power (highest score)
        4: 2.0   # identity (lowest score)
    }
    
    for i, state in enumerate(states):
        total_score = 0.0
        function_counts = {idx: 0 for idx in function_scores.keys()}
        
        # Count occurrences of each function
        for layer in state:
            for func_idx in layer:
                if func_idx in function_scores:
                    function_counts[func_idx] += 1
        
        # Calculate weighted score
        for func_idx, count in function_counts.items():
            if count > 0:
                # Base score for having the function
                base_score = function_scores[func_idx]
                # Bonus for multiple instances (diminishing returns)
                bonus = torch.log1p(torch.tensor(count)).item()
                total_score += base_score * bonus
        
        # Add diversity bonus if using multiple different functions
        num_different_functions = sum(1 for count in function_counts.values() if count > 0)
        diversity_bonus = num_different_functions * 0.5
        
        # Compute final reward with base score and bonuses
        rewards[i] = torch.tensor(1.0 + total_score + diversity_bonus)
        
        # Print detailed scoring for best rewards (optional)
        if rewards[i] > 10.0:  # Adjust threshold as needed
            print(f"\nHigh reward structure found ({rewards[i]:.2f}):")
            for func_idx, count in function_counts.items():
                if count > 0:
                    print(f"Function {func_idx}: {count} instances")
            print(f"Diversity bonus: {diversity_bonus:.2f}")
            print("-" * 30)
    
    return rewards


if __name__ == "__main__":
    main()