import torch
import sys
import os
from pathlib import Path

# Ensure the project root is in the system path
def add_project_root_to_sys_path():
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

add_project_root_to_sys_path()

from src.models.custom_functions import SafeExp, SafeLog, SafeSin, SafePower, SafeIdentityFunction

class Functions:
    def __init__(self):
        self.functions = {
            "exp": SafeExp(),
            "log": SafeLog(),
            "sin": SafeSin(),
            "power": SafePower(),
            "identity": SafeIdentityFunction()
            # Idea: Add "x" function just to know x in the layer
        }

    def get_function(self, function_name):
        function_class = self.functions.get(function_name)
        if function_class is None:
            raise ValueError(f"Unknown function: {function_name}")
        return function_class()



def get_next_node_indices(neural_network_states, placeholder_unary=-1, placeholder_binary=-2):
    """
    Determine the indices of the next nodes to be assigned in each neural network state.
    
    Args:
        neural_network_states: Batch of neural network states as tensors (batch_size, max_layers, max_nodes_per_layer)
                               where placeholders represent unassigned nodes
        placeholder_unary: Value used for unary function placeholder (-1 by default)
        placeholder_binary: Value used for binary function placeholder (-2 by default)
                            
    Returns:
        nodes_to_assign: Tuple of (layer_idx, node_idx) for each state, identifying the next node to assign
        prev_nodes: Values of previous nodes in the same layer (or placeholder if not available)
        layer_info: Information about the layer containing the node to be assigned
    """
    batch_size = neural_network_states.shape[0]
    
    # Initialize return values
    layer_indices = torch.zeros(batch_size, dtype=torch.long)
    node_indices = torch.zeros(batch_size, dtype=torch.long)
    prev_nodes = torch.ones(batch_size, dtype=torch.long) * -999  # Default value if no previous node
    layer_info = torch.zeros(batch_size, dtype=torch.long)  # Will store layer size or other relevant info
    
    # For each state in the batch
    for b in range(batch_size):
        found = False
        # Iterate through layers
        for layer_idx in range(neural_network_states.shape[1]):
            # Iterate through nodes in the layer
            for node_idx in range(neural_network_states.shape[2]):
                # Check if this node is a placeholder (unassigned)
                if (neural_network_states[b, layer_idx, node_idx] == placeholder_unary or 
                    neural_network_states[b, layer_idx, node_idx] == placeholder_binary):
                    # Record the layer and node indices
                    layer_indices[b] = layer_idx
                    node_indices[b] = node_idx
                    
                    # Get previous node value if available
                    if node_idx > 0:
                        prev_nodes[b] = neural_network_states[b, layer_idx, node_idx-1]
                    
                    # Store layer info (e.g., number of nodes in the layer)
                    non_placeholder_mask = (neural_network_states[b, layer_idx] != placeholder_unary) & \
                                           (neural_network_states[b, layer_idx] != placeholder_binary)
                    layer_info[b] = non_placeholder_mask.sum()
                    
                    found = True
                    break
            if found:
                break
    
    # Combine layer and node indices
    nodes_to_assign = (layer_indices, node_indices)
    
    return nodes_to_assign, prev_nodes, layer_info


def convert_list_structure_to_tensor(neural_network_states, max_layers=None, max_nodes=None, padding_value=-999):
    """
    Convert a list of list structure representing neural networks to a tensor format.
    
    Args:
        neural_network_states: List of neural network states where each state is a list of layers,
                               and each layer is a list of node function assignments
        max_layers: Maximum number of layers to include (pad or truncate)
        max_nodes: Maximum number of nodes per layer (pad or truncate)
        padding_value: Value to use for padding
        
    Returns:
        torch.Tensor: Tensor representation of the neural network states
    """
    batch_size = len(neural_network_states)
    
    # Determine the maximum dimensions if not provided
    if max_layers is None:
        max_layers = max(len(state) for state in neural_network_states)
    
    if max_nodes is None:
        max_nodes = max(max(len(layer) for layer in state) for state in neural_network_states)
    
    # Create a padded tensor
    tensor_states = torch.ones(batch_size, max_layers, max_nodes, dtype=torch.long) * padding_value
    
    # Fill in the tensor with the state values
    for b, state in enumerate(neural_network_states):
        for l, layer in enumerate(state):
            if l < max_layers:
                for n, node_value in enumerate(layer):
                    if n < max_nodes:
                        tensor_states[b, l, n] = node_value
    
    return tensor_states


def convert_tensor_to_list_structure(tensor_states, padding_value=-999):
    """
    Convert a tensor representation of neural network states back to the list of lists structure.
    
    Args:
        tensor_states: Tensor representation of neural network states
        padding_value: Value used for padding in the tensor
        
    Returns:
        list: Neural network states as a list of lists
    """
    batch_size, max_layers, max_nodes = tensor_states.shape
    
    # Convert back to list structure
    neural_network_states = []
    
    for b in range(batch_size):
        state = []
        for l in range(max_layers):
            layer = []
            for n in range(max_nodes):
                value = tensor_states[b, l, n].item()
                if value != padding_value:
                    layer.append(value)
            if layer:  # Only add the layer if it has any nodes
                state.append(layer)
        neural_network_states.append(state)
    
    return neural_network_states


def mask_and_normalize(states, num_functions, function_categories, 
                       placeholder_unary=-1, placeholder_binary=-2, padding_value=-999):
    """
    Create a mask over the function space based on the current neural network state
    and normalize the probabilities accordingly.
    
    Args:
        states: Batch of neural network states (either as list of lists or tensor)
        num_functions: Total number of available functions
        function_categories: Dictionary mapping function types to their indices
                            (e.g., {'activation': [0,1,2], 'linear': [3,4], 'special': [5,6]})
        placeholder_unary: Value used for unary function placeholder
        placeholder_binary: Value used for binary function placeholder
        padding_value: Value used for padding
        
    Returns:
        torch.Tensor: Mask tensor (batch_size, num_functions) where 1 indicates allowed functions
        torch.Tensor: Boolean tensor indicating which states are completed (fully assigned)
    """
    # Convert to tensor if not already
    if not isinstance(states, torch.Tensor):
        tensor_states = convert_list_structure_to_tensor(states, padding_value=padding_value)
    else:
        tensor_states = states
        
    batch_size = tensor_states.shape[0]
    
    # Initialize mask with all functions allowed
    mask = torch.ones((batch_size, num_functions))
    
    # Get the next node to assign for each state
    nodes_to_assign, prev_nodes, layer_info = get_next_node_indices(tensor_states, 
                                                                placeholder_unary, 
                                                                placeholder_binary)
    layer_indices, node_indices = nodes_to_assign
    
    # Identify completed networks (no nodes left to assign)
    any_placeholder = ((tensor_states == placeholder_unary) | (tensor_states == placeholder_binary))
    done_idx = ~any_placeholder.any(dim=(1, 2))
    
    # For completed networks, set all functions to invalid except "done" function if it exists
    mask[done_idx, :] = 0
    if 'done' in function_categories:
        mask[done_idx, function_categories['done']] = 1
    
    # Apply rule 1: Input layer can only use certain types of functions (e.g., linear)
    is_input_layer = layer_indices == 0
    if 'input_only' in function_categories:
        # Disable all functions for input layer
        mask[is_input_layer, :] = 0
        # Enable only input-specific functions
        mask[is_input_layer, function_categories['input_only']] = 1
    
    # Apply rule 2: Output layer can only use certain types of functions
    max_layer_idx = tensor_states.shape[1] - 1
    is_output_layer = layer_indices == max_layer_idx
    if 'output_only' in function_categories:
        # Disable all functions for output layer
        mask[is_output_layer, :] = 0
        # Enable only output-specific functions
        mask[is_output_layer, function_categories['output_only']] = 1
    
    # Apply rule 3: If previous node in same layer has certain functions, restrict current node
    if 'activation' in function_categories and 'linear' in function_categories:
        # Get indices where previous node has a linear function
        prev_is_linear = torch.zeros(batch_size, dtype=torch.bool)
        for i in range(batch_size):
            if prev_nodes[i] in function_categories['linear']:
                prev_is_linear[i] = True
        
        # If previous node is linear, current node can only be activation
        mask[prev_is_linear, :] = 0
        mask[prev_is_linear, function_categories['activation']] = 1
    
    # Apply rule 4: Hidden layers can use all function types except those restricted to input/output
    is_hidden_layer = ~(is_input_layer | is_output_layer)
    if 'hidden_only' in function_categories:
        # For hidden layers, enable all functions except those in 'input_only' or 'output_only'
        restricted_funcs = []
        if 'input_only' in function_categories:
            restricted_funcs.extend(function_categories['input_only'])
        if 'output_only' in function_categories:
            restricted_funcs.extend(function_categories['output_only'])
        
        for i in range(batch_size):
            if is_hidden_layer[i]:
                mask[i, restricted_funcs] = 0
                mask[i, function_categories['hidden_only']] = 1
    
    # Ensure we don't completely mask out all functions for any unfinished state
    for i in range(batch_size):
        if not done_idx[i] and mask[i].sum() == 0:
            # If everything is masked, just allow all functions
            mask[i] = 1
    
    return mask, done_idx


def get_default_function_categories():
    """
    Create a default dictionary of function categories for symbolic regression with neural networks.
    This categorizes different functions based on their role and where they can be used in the network.
    
    Returns:
        dict: A dictionary mapping function types to their indices
    """
    # Define function indices based on the listing in the Functions class
    # These indices should match the order of functions in your implementation
    function_categories = {
        # Activation functions
        'activation': [0, 1, 2],  # e.g., sigmoid, tanh, relu
        
        # Linear functions
        'linear': [3, 4],  # e.g., identity, linear
        
        # Functions suitable for input layer
        'input_only': [3, 4],  # Typically linear functions
        
        # Functions suitable for hidden layers
        'hidden_only': [0, 1, 2, 3, 4, 5, 6],  # All functions
        
        # Functions suitable for output layer
        'output_only': [0, 1, 2, 3, 4],  # Activation and linear functions
        
        # Special functions that can be used anywhere
        'special': [5, 6]  # e.g., constant, power functions
    }
    
    return function_categories


def get_function_categories_from_names(function_names, function_types):
    """
    Create a function categories dictionary based on function names and their types.
    
    Args:
        function_names: List of function names in order of their indices
        function_types: Dictionary mapping function types to lists of function names
        
    Returns:
        dict: A dictionary mapping function types to their indices
    """
    function_categories = {}
    
    # Create a mapping from function name to index
    name_to_idx = {name: idx for idx, name in enumerate(function_names)}
    
    # For each function type, find the corresponding indices
    for func_type, func_names in function_types.items():
        function_categories[func_type] = [name_to_idx[name] for name in func_names if name in name_to_idx]
    
    return function_categories
