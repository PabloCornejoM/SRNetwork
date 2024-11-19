import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy
import numpy as np

class BaseSafeFunction(nn.Module):
    """Base class for safe mathematical functions with weight initialization."""
    
    def __init__(self, name, sympy_repr):
        super().__init__()
        self.name = name
        self.sympy_repr = sympy_repr
        self.weight = None
        self.bias = None

    def init_parameters(self, input_size, output_size):
        """Initialize weights and biases with appropriate dimensions."""
        self.weight = nn.Parameter(torch.empty(input_size, output_size))
        self.bias = nn.Parameter(torch.zeros(output_size))
        self._init_weight_values()
        self._init_bias_values()
    
    def _init_weight_values(self):
        """Default weight initialization."""
        nn.init.xavier_uniform_(self.weight)
    
    def _init_bias_values(self):
        """Default bias initialization."""
        nn.init.zeros_(self.bias)

class SafeExp(BaseSafeFunction):
    """Safe exponential implementation with controlled scaling."""
    
    def __init__(self, max_val=30):
        super().__init__("exp", "exp")
        self.max_val = max_val

    def forward(self, x):
        #x = torch.matmul(x, self.weight) + self.bias
        #return 1 / torch.exp(torch.clamp(x, min=-self.max_val, max=self.max_val))
        return torch.exp(torch.clamp(x, min=-self.max_val, max=self.max_val))
    
    
    def _init_weight_values(self):
        # Initialize with negative identity matrices as in your implementation
        times = int(self.weight.shape[1] // self.weight.shape[0]) + 1
        output_matrix = [-1 * (i+1) * np.eye(self.weight.shape[0]) for i in range(times)]
        output_matrix = np.concatenate(output_matrix, axis=-1)
        output_matrix = torch.Tensor(output_matrix[:, :self.weight.shape[1]])
        self.weight.data.copy_(output_matrix)

class SafeLog(BaseSafeFunction):
    """Safe logarithm implementation with proper handling of small/negative values."""
    
    def __init__(self, eps=1e-3):
        super().__init__("log", "log")
        self.eps = eps

    def forward(self, x):
        #x = torch.matmul(x, self.weight) + self.bias
        x = x.double()
        return torch.where(
            torch.abs(x) > self.eps,
            torch.log(torch.abs(x)),
            0.
        ).float()
    
    def _init_weight_values(self):
        # Initialize with random signs as in your implementation
        signs = torch.where(
            torch.rand_like(self.weight) > 0.5,
            torch.ones_like(self.weight),
            -torch.ones_like(self.weight)
        )
        self.weight.data.copy_(signs)
    
    def _init_bias_values(self):
        signs = torch.where(
            torch.rand_like(self.bias) > 0.5,
            torch.ones_like(self.bias),
            -torch.ones_like(self.bias)
        )
        self.bias.data.copy_(signs)

class SafeSin(BaseSafeFunction):
    """Safe sine implementation with controlled scaling."""
    
    def __init__(self):
        super().__init__("sin", "sin")

    def forward(self, x):
        x = torch.matmul(x, self.weight) + self.bias
        return torch.sin(x)
    
    def _init_weight_values(self):
        signs = torch.where(
            torch.rand_like(self.weight) > 0.5,
            torch.ones_like(self.weight),
            -torch.ones_like(self.weight)
        )
        self.weight.data.copy_(signs)
    
    def clip(self):
        """Clip weights and biases to controlled range."""
        torch.clip_(self.weight, -1, 1)
        torch.clip_(self.bias, -1, 1)

# Map PyTorch functions to their sympy equivalents
SYMPY_MAPPING = {
    SafeLog: sympy.log,
    SafeExp: sympy.exp,
    SafeSin: sympy.sin
} 