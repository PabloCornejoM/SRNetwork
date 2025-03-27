import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.distributions import Categorical
from log import Log
from functions import mask_and_normalize as custom_mask_and_normalize


class GFlowNet(nn.Module):
    def __init__(self, forward_policy, backward_policy, env):
        """
        Initializes a GFlowNet using the specified forward and backward policies
        acting over an env, i.e. a state space and a reward function.
        
        Args:
            forward_policy: A policy network taking as input a state and
            outputting a vector of probabilities over actions
            
            backward_policy: A policy network (or fixed function) taking as
            input a state and outputting a vector of probabilities over the
            actions which led to that state
            
            env: An env defining a state space and an associated reward
            function
        """
        super().__init__()
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy
        self.env = env
        # the total_flow is a learnable param
        self.total_flow = Parameter(torch.ones(1))
    
    def mask_and_normalize(self, s, probs):
        """
        Masks a vector of action probabilities to avoid illegal actions (i.e.
        actions that lead outside the state space).
        
        Args:
            s: Batch of neural network states (list of lists representing layers)
            
            probs: An NxA matrix of action probabilities
        """
        # 1e-8 for smoothing and avoiding division by zero
        if hasattr(self.env, 'mask'):
            # Use the environment's mask if available
            mask, done_idx = self.env.mask(s)
        else:
            # Otherwise use our custom masking function
            mask, done_idx = custom_mask_and_normalize(
                s, 
                self.forward_policy.num_functions, 
                self.forward_policy.function_categories,
                self.forward_policy.placeholder_unary,
                self.forward_policy.placeholder_binary
            )
        
        # Apply mask and normalize probabilities
        probs = mask * (probs + 1e-8)
        probs = probs / probs.sum(1).unsqueeze(1)
        
        return probs, done_idx
        
    def forward_probs(self, s):
        """
        Returns a vector of probabilities over actions in a given state.
        
        Args:
            s: Batch of neural network states (list of lists representing layers)
        """
        # The forward_policy now handles masking internally
        probs = self.forward_policy(s, apply_mask=False)
        return self.mask_and_normalize(s, probs)
    
    def sample_states(self, s0):
        """
        Samples and returns a collection of final states from the GFlowNet.
        
        Args:
            s0: Initial states (list of lists representing neural network layers with placeholders)
        """
        # Use the forward policy's sample_states method which already implements the full sampling logic
        states, log_probs, action_sequences = self.forward_policy.sample_states(
            s0, 
            max_steps=100, 
            apply_mask=False
        )
        
        # Calculate rewards for sampled states
        #rewards = self.env.reward(states)
        
        # Create a log object to track information about the sampling process
        #log = Log(states, log_probs, rewards, self.total_flow)
        
        return states, log_probs, action_sequences, self.total_flow #log
    
    def evaluate_trajectories(self, traj, actions):
        """
        Returns the GFlowNet's estimated forward probabilities, backward
        probabilities, and rewards for a collection of trajectories. This is
        useful in an offline learning context where samples drawn according to
        another policy (e.g. a random one) are used to train the model.
        
        Args:
            traj: The trajectory of each sample
            
            actions: The actions that produced the trajectories in traj
        """
        # This method would need to be reimplemented for neural networks
        # but it doesn't appear to be used in the current training workflow
        raise NotImplementedError("evaluate_trajectories is not yet implemented for neural networks")


def trajectory_balance_loss(total_flow, rewards, fwd_probs):
    """
    Compute the trajectory balance loss as described in Bengio et al. (2022).
    The loss is computed as log squared difference between the left-hand side
    (total_flow * prod(fwd_probs)) and the rewards.
    
    Args:
        total_flow: The total flow for each trajectory
        rewards: The rewards associated with the final state of each trajectory
        fwd_probs: The forward probabilities associated with each node in the trajectory
                  (shape: [batch_size, num_nodes])
    """
    # Calculate left-hand side of the trajectory balance equation
    # Sum log probabilities across nodes and take exp to get product
    lhs = total_flow * torch.exp(torch.sum(fwd_probs, dim=1))
    
    # Calculate the loss as log squared difference
    loss = torch.log(lhs / rewards)**2
    
    # Check for numerical issues
    assert torch.isfinite(loss).all(), total_flow
    
    return loss.mean()