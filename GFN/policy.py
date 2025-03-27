import torch
from torch import nn
import torch.nn.functional as F
from functions import get_next_node_indices, mask_and_normalize

"""
All the forward policies for GFlowNet. The `RNNForwardPolicy` contains implementations using vanilla RNN,
GRU, and LSTM. The `CanonicalBackwardPolicy` serves as a placeholder since the backward probabilities is
trivial when the state space has a tree structure.
"""


class RNNForwardPolicy(nn.Module):
    def __init__(self, batch_size, hidden_dim, num_functions,
                 num_layers=1, model='rnn', dropout=0.0, placeholder_unary=-1, 
                 placeholder_binary=-2, device=None, function_categories=None):
        super(RNNForwardPolicy, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_dim
        self.num_functions = num_functions  # Number of available functions (unary + binary)
        self.num_layers = num_layers
        self.dropout = dropout
        self.placeholder_unary = placeholder_unary
        self.placeholder_binary = placeholder_binary
        self.device = torch.device("cpu") if not device else device
        self.model = model
        self.function_categories = function_categories or {}

        # Input dimension: 5 features per node
        # 1. Type (-1 for unary, -2 for binary)
        # 2. Position in layer
        # 3. Layer number
        # 4. Previous node value in same layer
        # 5. Next node value in same layer
        state_dim = 5

        if model == 'rnn':
            self.rnn = nn.RNN(state_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=self.dropout).to(self.device)
        elif model == 'gru':
            self.rnn = nn.GRU(state_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=self.dropout).to(self.device)
        elif model == 'lstm':
            self.rnn = nn.LSTM(state_dim, hidden_dim, num_layers,
                             batch_first=True, dropout=self.dropout).to(self.device)
            self.init_c0 = torch.zeros(self.num_layers, self.hidden_size).to(self.device)
        else:
            raise NotImplementedError("unsupported model: " + model)

        self.fc = nn.Linear(hidden_dim, num_functions).to(self.device)
        self.init_h0 = torch.zeros(self.num_layers, self.hidden_size).to(self.device)

    def get_next_node_info(self, state):
        """
        Determine the next node to be assigned a function in the neural network.
        
        Args:
            state: List of lists where each inner list represents a layer in the neural network
                  with node values (-1 for unary placeholder, -2 for binary placeholder, >=0 for assigned functions)
        
        Returns:
            tuple: (layer_idx, node_idx, node_type) or (None, None, None) if all nodes are assigned
        """
        # Iterate through layers
        for layer_idx, layer in enumerate(state):
            # Iterate through nodes in the layer
            for node_idx, node_value in enumerate(layer):
                # If node is a placeholder (unassigned), return its information
                if node_value == self.placeholder_unary:
                    return layer_idx, node_idx, self.placeholder_unary
                elif node_value == self.placeholder_binary:
                    return layer_idx, node_idx, self.placeholder_binary
        
        # If all nodes are assigned, return None values
        return None, None, None

    def forward(self, states, apply_mask=True):
        """
        Forward pass to get action probabilities for the next node to be assigned.
        
        Args:
            states: Batch of current states (list of lists representing neural network layers)
            apply_mask: Whether to apply masking rules to the probabilities
            
        Returns:
            torch.Tensor: Probabilities over possible functions
        """
        batch_size = len(states)
        
        # Initialize or reset hidden state for new sequences
        # Check if we're at the beginning state (first node has placeholder)
        if any(state[0][0] in [self.placeholder_unary, self.placeholder_binary] for state in states):
            self.h0 = self.init_h0.unsqueeze(1).repeat(1, batch_size, 1)
            if self.model == 'lstm':
                self.c0 = self.init_c0.unsqueeze(1).repeat(1, batch_size, 1)

        # Process each state in the batch
        encodings = []
        for state in states:
            # Get information about the next node to assign
            layer_idx, node_idx, node_type = self.get_next_node_info(state)
            
            if layer_idx is not None:
                # Get context information (adjacent nodes)
                prev_node_value = state[layer_idx][node_idx-1] if node_idx > 0 else -999
                next_node_value = state[layer_idx][node_idx+1] if node_idx < len(state[layer_idx])-1 else -999
                
                # Create full encoding for the node with context
                encoding = torch.tensor([
                    float(node_type),       # Node type
                    float(node_idx),        # Position in layer
                    float(layer_idx),       # Layer number
                    float(prev_node_value), # Previous node in same layer
                    float(next_node_value), # Next node in same layer
                ], device=self.device).unsqueeze(0).unsqueeze(0)
                
                encodings.append(encoding)
            else:
                # If no nodes to assign, use a default encoding
                default_encoding = torch.zeros(1, 1, 5, device=self.device)
                encodings.append(default_encoding)

        # Stack all encodings in the batch
        rnn_input = torch.cat(encodings, dim=0)

        # RNN forward pass
        if self.model == 'lstm':
            output, (self.h0, self.c0) = self.rnn(rnn_input, (self.h0, self.c0))
        else:
            output, self.h0 = self.rnn(rnn_input, self.h0)

        # Get raw logits
        logits = self.fc(output[:, -1, :])
        
        # If masking is enabled, apply constraints to the probabilities
        if apply_mask and self.function_categories:
            # Create mask and get done indices
            mask, done_idx = mask_and_normalize(states, self.num_functions, 
                                               self.function_categories,
                                               self.placeholder_unary, 
                                               self.placeholder_binary)
            
            # Apply mask by setting invalid logits to a large negative value
            mask = mask.to(self.device)
            masked_logits = logits.clone()
            masked_logits[mask == 0] = float('-inf')
            
            # Compute probabilities from masked logits
            probabilities = F.softmax(masked_logits, dim=1)
        else:
            # Compute probabilities without masking
            probabilities = F.softmax(logits, dim=1)

        return probabilities.cpu()
    
    def sample_states(self, initial_states, max_steps=100, apply_mask=True):
        """
        Generate complete neural network architectures by progressively sampling functions for each node.
        
        Args:
            initial_states: Batch of initial states (list of lists representing neural network layers with placeholders)
            max_steps: Maximum number of steps to sample to prevent infinite loops
            apply_mask: Whether to apply masking rules to the probabilities
            
        Returns:
            list: Batch of final states with all nodes assigned functions
            torch.Tensor: Log probabilities of each sampled state (shape: [batch_size, num_nodes])
            list: Sequence of actions taken to construct each state
        """
        batch_size = len(initial_states)
        
        # Create deep copies of the initial states to avoid modifying originals
        states = []
        for state in initial_states:
            copied_state = []
            for layer in state:
                # Handle both tensor and list inputs
                if isinstance(layer, torch.Tensor):
                    copied_state.append(layer.clone())
                else:
                    copied_state.append(layer.copy())
            states.append(copied_state)
        
        # Count total number of nodes that need to be assigned
        total_nodes = sum(len(layer) for state in initial_states for layer in state)
        
        # Initialize log probabilities tensor with shape [batch_size, total_nodes]
        log_probs = torch.zeros(batch_size, total_nodes)
        action_sequences = [[] for _ in range(batch_size)]
        
        steps = 0
        all_assigned = False
        node_idx = 0
        
        while not all_assigned and steps < max_steps:
            # Check if states are fully assigned using the done_idx from masking
            if apply_mask and self.function_categories:
                _, done_idx = mask_and_normalize(states, self.num_functions, 
                                               self.function_categories,
                                               self.placeholder_unary, 
                                               self.placeholder_binary)
                all_assigned = done_idx.all().item()
            else:
                # Use the original method to check if all nodes are assigned
                all_assigned = True
                for state in states:
                    layer_idx, node_idx, _ = self.get_next_node_info(state)
                    if layer_idx is not None:
                        all_assigned = False
                        break
            
            if all_assigned:
                break
                
            # Get probabilities for next function assignment
            probs = self.forward(states, apply_mask=apply_mask)
            
            # Sample actions for each state in the batch
            actions = torch.multinomial(probs, 1).squeeze(1)
            
            # Update states with sampled actions
            for i, (state, action) in enumerate(zip(states, actions)):
                layer_idx, node_idx, _ = self.get_next_node_info(state)
                if layer_idx is not None:
                    # Update the state with the sampled function
                    state[layer_idx][node_idx] = action.item()
                    
                    # Update log probability for this node
                    log_probs[i, node_idx] = torch.log(probs[i, action])
                    action_sequences[i].append(action.item())
            
            steps += 1
        
        return states, log_probs, action_sequences




class CanonicalBackwardPolicy(nn.Module):
    def __init__(self, num_functions: int):
        super(CanonicalBackwardPolicy, self).__init__()
        self.num_functions = num_functions

    def forward(self, states):
        """
        Calculate the backward probability matrix for a given state.
        For neural network architectures, we need to find the last assigned function
        in the layer-wise representation.
        
        Args:
            states: List of lists where each sublist represents a layer
                   containing function assignments and placeholders (-1, -2)
        
        Returns:
            probs: a (batch_size x num_functions) probability matrix
        """
        batch_size = len(states)
        last_actions = []

        for state in states:
            # Convert state to tensor for easier processing
            state_tensor = torch.tensor([item for layer in state for item in layer])
            # Find indices where we have actual functions (>= 0)
            valid_indices = (state_tensor >= 0).nonzero()
            
            if len(valid_indices) == 0:
                # If no actions have been taken yet, this shouldn't happen in practice
                last_actions.append(0)
            else:
                # Get the last assigned function
                last_action = state_tensor[valid_indices[-1]]
                last_actions.append(last_action)

        # Convert to tensor and create one-hot encoding
        actions_tensor = torch.tensor(last_actions)
        probs = F.one_hot(actions_tensor, self.num_functions).float()
        
        return probs
