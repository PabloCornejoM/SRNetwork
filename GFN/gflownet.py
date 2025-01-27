import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from tqdm import tqdm  # Import tqdm for progress tracking

import sys
from pathlib import Path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)


from custom_functions import (SafeIdentityFunction, SafeLog, 
                                SafeExp, SafeSin, SafePower)

# Dictionary of functions
functions = {
    'id': SafeIdentityFunction,
    'log': SafeLog,
    'sin': SafeSin,
}

# Sort the keys
sorted_keys = sorted(functions.keys())
print(sorted_keys)

# We first define how the model will view a face, i.e. how to encode a face in
# a tensor
def function_to_tensor(functions):
    return torch.tensor([i in functions for i in sorted_keys]).float()

class FlowModel(nn.Module):
    def __init__(self, num_hid):
        super().__init__()
        # We encoded the current state as binary vector, for each patch the associated
        # dimension is either 0 or 1 depending on the absence or presence of that patch.
        # Therefore the input dimension is 3 for the 3 actions.
        self.mlp = nn.Sequential(
            nn.Linear(3, num_hid),
            nn.LeakyReLU(),
            nn.Linear(num_hid, 3)  # Output 3 numbers for possible actions
        )

    def forward(self, x):
        # We take the exponential to get positive numbers, since flows must be positive,
        # and multiply by (1 - x) to give 0 flow to actions we know we can't take
        F = self.mlp(x).exp()  #* (1 - x)
        return F

def face_parents(state):
    parent_states = []  # states that are parents of state
    parent_actions = []  # actions that lead from those parents to state
    for face_part in state:
        # For each face part, there is a parent without that part
        parent_states.append([i for i in state if i != face_part])
        # The action to get there is the corresponding index of that face part
        parent_actions.append(sorted_keys.index(face_part))
    return parent_states, parent_actions

