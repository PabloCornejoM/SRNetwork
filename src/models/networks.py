import torch
import torch.nn as nn
from typing import Dict, Any

class BaseModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.build_model()
        
    def build_model(self):
        """
        Initialize model architecture.
        Should be implemented by child classes.
        """
        raise NotImplementedError
        
    def forward(self, x):
        """
        Forward pass of the model.
        Should be implemented by child classes.
        """
        raise NotImplementedError

# Your specific model implementations here 