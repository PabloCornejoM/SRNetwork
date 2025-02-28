import torch
from torch import nn
import torch.nn.functional as F
from actions import get_next_node_indices

"""
All the forward policies for GFlowNet. The `RNNForwardPolicy` contains implementations using vanilla RNN,
GRU, and LSTM. The `CanonicalBackwardPolicy` serves as a placeholder since the backward probabilities is
trivial when the state space has a tree structure.
"""


class RNNForwardPolicy(nn.Module):
    def __init__(self, batch_size, hidden_dim, num_functions,
                 num_layers=1, model='rnn', dropout=0.0, placeholder_unary=-1, 
                 placeholder_binary=-2, device=None):
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

        # Input dimension: 3 features per node
        # 1. Type (-1 for unary, -2 for binary)
        # 2. Position in layer
        # 3. Layer number
        state_dim = 3

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
        Find the next node to be assigned and its context information.
        
        Args:
            state: List of lists representing the current state of each layer
            
        Returns:
            tuple: (layer_idx, node_idx, node_type)
        """
        for layer_idx, layer in enumerate(state):
            for node_idx, node_type in enumerate(layer):
                if node_type in [self.placeholder_unary, self.placeholder_binary]:
                    return layer_idx, node_idx, node_type
        return None, None, None

    def create_node_encoding(self, layer_idx, node_idx, node_type):
        """
        Create the encoding for a node to be fed into the RNN.
        
        Args:
            layer_idx: Index of the layer
            node_idx: Index of the node in the layer
            node_type: Type of the node (-1 for unary, -2 for binary)
            
        Returns:
            torch.Tensor: Encoded node features
        """
        encoding = torch.tensor([
            float(node_type),  # Node type
            float(node_idx),   # Position in layer
            float(layer_idx)   # Layer number
        ], device=self.device).unsqueeze(0).unsqueeze(0)
        
        return encoding

    def forward(self, states):
        """
        Forward pass to get action probabilities for the next node to be assigned.
        
        Args:
            states: Batch of current states (list of lists representing layer structures)
            
        Returns:
            torch.Tensor: Probabilities over possible functions
        """
        batch_size = len(states)
        
        # Initialize or reset hidden state for new sequences
        if isinstance(states[0][0][0], (int, float)) and states[0][0][0] in [self.placeholder_unary, self.placeholder_binary]:
            self.h0 = self.init_h0.unsqueeze(1).repeat(1, batch_size, 1)
            if self.model == 'lstm':
                self.c0 = self.init_c0.unsqueeze(1).repeat(1, batch_size, 1)

        # Process each state in the batch
        encodings = []
        for state in states:
            layer_idx, node_idx, node_type = self.get_next_node_info(state)
            if layer_idx is not None:
                encoding = self.create_node_encoding(layer_idx, node_idx, node_type)
                encodings.append(encoding)

        # Stack all encodings in the batch
        rnn_input = torch.cat(encodings, dim=0)

        # RNN forward pass
        if self.model == 'lstm':
            output, (self.h0, self.c0) = self.rnn(rnn_input, (self.h0, self.c0))
        else:
            output, self.h0 = self.rnn(rnn_input, self.h0)

        # Get function probabilities
        output = self.fc(output[:, -1, :])
        probabilities = F.softmax(output, dim=1)

        return probabilities.cpu()
    



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
