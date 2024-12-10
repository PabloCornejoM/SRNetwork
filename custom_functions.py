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
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
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

class SafePower(BaseSafeFunction):
    """
    Safe power function with trainable exponents and sign handling.
    Implements x^p where p is trainable and handles negative inputs safely.
    """
    def __init__(self):
        super().__init__("power", "pow")
        self.hardsigmoid = nn.Hardsigmoid()
        self.sign_params = None

    def init_parameters(self, input_size, output_size):
        """Initialize parameters for EQL layer integration"""
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.zeros(output_size))
        self.sign_params = nn.Parameter(torch.zeros(output_size))
        
        # Initialize exponents between 1 and 3
        nn.init.uniform_(self.weight, 1.0, 3.0)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.sign_params)

    def forward(self, x):
        # Handle signs
        input_signs = torch.sign(x)
        x_abs = torch.clamp(torch.abs(x), min=1e-7)
        
        # Compute power using log-exp trick for numerical stability
        log_x = torch.log(x_abs)
        power_result = torch.exp(torch.matmul(log_x, self.weight.t()))
        
        # Determine output sign based on input sign and learned sign behavior
        sign_factors = self.hardsigmoid(self.sign_params)  # Shape: scalar or (1)
        if not self.training:
            sign_factors = (sign_factors > 0.5).float()
        
        # Compute the sign tensor similar to the first function
        sign = torch.ones_like(input_signs) * sign_factors + input_signs * (1 - sign_factors)  # Shape: (n, d)
        sign = torch.prod(sign, dim=1, keepdim=True)  # Shape: (n, 1)
        
        # Multiply the sign with the power result
        power_out = sign * power_result + self.bias  # Shapes: (n, 1) * (n, output_dim) => (n, output_dim)
        
        return power_out

    def clip_parameters(self):
        """Clip parameters to reasonable ranges"""
        with torch.no_grad():
            self.weight.data.clamp_(-4.0, 6.0)

    def get_sympy_expression(self, x, weight, sign):
        """Helper method to generate sympy expression for symbolic representation"""
        # Create a power expression with the weight as the exponent
        # and handle the sign based on the sign parameter
        if sign > 0.5:  # even power behavior
            return sympy.Abs(x)**weight
        else:  # odd power behavior
            return sympy.sign(x) * (sympy.Abs(x)**weight)

# Update the SYMPY_MAPPING dictionary to include SafePower
def power_to_sympy(x, **kwargs):
    """Convert power function to sympy expression"""
    weight = kwargs.get('weight', 1)
    sign = kwargs.get('sign', 0)
    if sign > 0.5:  # even power behavior
        return sympy.Abs(x)**weight
    else:  # odd power behavior
        return sympy.sign(x) * (sympy.Abs(x)**weight)

SYMPY_MAPPING = {
    SafeLog: sympy.log,
    SafeExp: sympy.exp,
    SafeSin: sympy.sin,
    SafePower: power_to_sympy  # Add SafePower to the mapping
} 