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
            "exp": SafeExp,
            "log": SafeLog,
            "sin": SafeSin,
            "power": SafePower,
            "identity": SafeIdentityFunction
            # Idea: Add "x" function just to know x in the layer
        }

    def get_function(self, function_name):
        function_class = self.functions.get(function_name)
        if function_class is None:
            raise ValueError(f"Unknown function: {function_name}")
        return function_class()



def get_next_node_indices(encodings, placeholder: int = -2):
    """
    A placeholder in `encodings` takes value -2, while uninitialized values use -1

    Args:
        encodings: a (M * T) encoding matrix
        placeholder: the value identifier for an initialized (but not assigned) node
    Returns:
        nodes_to_assign: a (M, ) vector of node indices to apply next action
        siblings: a (M, ) vector of siblings of `nodes_to_assign`, 0 if not exist
        parents: a (M, ) vector of parents of `nodes_to_assign`, 0 if not exist
    """
    batch_size, _ = encodings.shape
    siblings = torch.ones(batch_size, dtype=torch.long) * placeholder

    # get the indices of most recent nodes (to be assigned value) for each sample
    # we assume that the tree cannot be fully initialized
    nodes_to_assign = (encodings == placeholder).long().argmax(axis=1)

    parent_idx = torch.div(nodes_to_assign - 1, 2, rounding_mode='floor').clamp(min=0)
    parents = encodings[torch.arange(batch_size), parent_idx]
    is_right_node = nodes_to_assign % 2 == 0
    sibling_idx = (nodes_to_assign[is_right_node] - 1).clamp(min=0)
    siblings[is_right_node] = encodings[is_right_node, sibling_idx]
    return nodes_to_assign, siblings, parents
