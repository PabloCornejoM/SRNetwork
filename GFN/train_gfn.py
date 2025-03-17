import torch
import argparse

import matplotlib.pyplot as plt
from gflownet import GFlowNet#, trajectory_balance_loss
from tqdm import tqdm
import sys
from pathlib import Path

# Ensure the project root is in the system path
def add_project_root_to_sys_path():
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

add_project_root_to_sys_path()


from functions import Functions

from src.utils.data_utils import get_nguyen_data_loaders, generate_nguyen_data
from src.models.eql_model import SRNetwork
from policy import RNNForwardPolicy, CanonicalBackwardPolicy



def main():
    # Initialize the function set
    functions = Functions().functions

    # Generate data from experiment
    train_loader, val_loader = get_nguyen_data_loaders('Nguyen-1', batch_size=64)
    
    # Get the full dataset for plotting
    X, y = generate_nguyen_data('Nguyen-1')

    # Initialize the environment
    input_size = X.shape[1]
    output_size = y.shape[1]
    batch_size = 1
    num_layers = 2
    nonlinear_info = [(3, 0), (0, 0), (0, 0)] # Network Structure
    env = SRNetwork(input_size, output_size, num_layers, functions, nonlinear_info) # Initialize the environment

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forward_policy = RNNForwardPolicy(batch_size, 250, env.num_actions, 1, model="lstm", device=device)
    backward_policy = CanonicalBackwardPolicy(env.num_actions)
    model = GFlowNet(forward_policy, backward_policy, env)
    params = [param for param in model.parameters() if param.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)
    flows, errs, avg_mses, top_mses = [], [], [], []


    for i in (p := tqdm(range(num_epochs=5000))):
        s0 = env.get_initial_states(batch_size)
        s, log = model.sample_states(s0)
        loss = trajectory_balance_loss(log.total_flow,
                                       log.rewards,
                                       log.fwd_probs)
        loss.backward()
        opt.step()
        opt.zero_grad()

        if i % 20 == 0:
            # avg_reward = log.rewards.mean().item()
            p.set_description(f"{loss.item():.3f}")
            flows.append(log.total_flow.item())
            errs.append(loss.item())
            avg_mse, top_mse = evaluate_model(env, model, eval_bs=100)
            avg_mses.append(avg_mse.item())
            top_mses.append(top_mse.item())

    # codes for plotting loss & rewards
    if show_plot:
        train_plot(errs, flows, avg_mses, top_mses)

    return model, env, errs, avg_mses, top_mses


def evaluate_model(env, model, eval_bs: int = 20, top_quantile: float = 0.1):
    eval_s0 = env.get_initial_states(eval_bs)
    eval_s, _ = model.sample_states(eval_s0)
    eval_mse = env.calc_loss(eval_s)
    eval_mse = eval_mse[torch.isfinite(eval_mse)]
    avg_mse = torch.median(eval_mse)
    top_mse = torch.quantile(eval_mse, q=top_quantile)
    return avg_mse, top_mse

    # Initialize the GFN
if __name__ == "__main__":
    main()