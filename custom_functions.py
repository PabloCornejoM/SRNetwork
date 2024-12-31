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
            torch.rand_like(self.weight) > 0.0,
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

            torch.rand_like(self.weight) > 0,
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
        self.count = 0

    def init_parameters(self, input_size, output_size):
        """Initialize parameters for EQL layer integration"""
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.zeros(output_size))
        self.sign_params = nn.Parameter(torch.zeros(output_size))
        
        # Initialize exponents between 1 and 6
        nn.init.uniform_(self.weight, 2+self.count, 2+self.count)
        #self.count += 1
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.sign_params)

    def forward(self, x, weight, sign_param):
        """
        Compute the forward pass for a learned power function: x^w.
        We blend between even and odd symmetry based on a learned 'sign_param'.
        
        Even function form: f(x) = exp(w * ln(|x|))
        Odd function form:  f(x) = sign(x) * exp(w * ln(|x|))
        
        Parameters:
        x          : Input tensor of shape (N, D)
        weight     : Learnable parameter for exponent, shape (out_dim, D)
        sign_param : Parameter controlling symmetry. After training:
                    sign_param > 0.5 => even function
                    sign_param <= 0.5 => odd function
        """

        # Ensure input magnitude is never zero (to avoid log(0))
        x_abs = torch.clamp(torch.abs(x), min=1e-7)
        input_signs = torch.sign(x)

        # Compute x^w using log-exp for stability:
        # log_x = ln(|x|)
        log_x = torch.log(x_abs)
        # power_result = exp(log_x * w), where w is from 'weight'
        # log_x: (N, D), weight: (out_dim, D), we do log_x @ weight.t(): (N, out_dim)
        power_result = torch.exp(log_x @ weight.t())  # shape: (N, out_dim)

        # Obtain a continuous factor from sign_param
        sign_factor = self.hardsigmoid(sign_param)  # in range [0,1]
        
        # During evaluation, we turn it into a hard binary decision:
        # >0.5 => even function, <=0.5 => odd function
        if not self.training:
            sign_factor = (sign_factor > 0.5).float()

        # sign_factor now represents how "even" we are:
        # If fully even (sign_factor=1): final_sign = 1
        # If fully odd (sign_factor=0): final_sign = sign(x)
        # If in between, we get a smooth mixture during training.
        final_sign = sign_factor * torch.ones_like(input_signs) + (1 - sign_factor) * input_signs

        # Compute final output
        # When even: f(x)=exp(w*ln|x|)
        # When odd:  f(x)=sign(x)*exp(w*ln|x|)
        # Mixture during training: a continuous blend.
        power_out = final_sign * power_result.unsqueeze(1) + self.bias

        return power_out

    def clip_parameters(self):
        """Clip parameters to reasonable ranges"""
        with torch.no_grad():
            self.weight.data.clamp_(-4.0, 6.0)

    def get_sympy_expression(self, x):
        """Helper method to generate sympy expression for symbolic representation"""
        # Create a power expression with the weight as the exponent
        # and handle the sign based on the sign parameter
        weight = self.weight.data[0].item()
        sign = self.sign_params.data[0].item()
        if sign > 0.5:  # even power behavior
            return f"{sympy.Abs(x)}^{weight}"
        else:  # odd power behavior
            return f"sign({x}) * {sympy.Abs(x)}^{weight}"

# Update the SYMPY_MAPPING dictionary to include SafePower
def power_to_sympy(x, **kwargs):
    """Convert power function to sympy expression"""
    weight = kwargs.get('weight', 1)
    sign = kwargs.get('sign', 0)
    
    # Create a visual representation of the power function
    if sign > 0.5:  # even power behavior
        return f"{sympy.Abs(x)}^{weight}"
    else:  # odd power behavior
        return f"sign({x}) * {sympy.Abs(x)}^{weight}"


class SafeIdentityFunction(BaseSafeFunction):
    """Base class for identity function."""
    
    def __init__(self):
        super().__init__("identity", "id")

    def forward(self, x):
        """Forward pass for identity function."""
        return x  # Simply return the input as is

    def _init_weight_values(self):
        """Identity function does not require weight initialization."""
        pass  # No weights to initialize for identity function

    def _init_bias_values(self):
        """Identity function does not require bias initialization."""
        pass  # No biases to initialize for identity function
 


SYMPY_MAPPING = {
    SafeLog: sympy.log,
    SafeExp: sympy.exp,
    SafeSin: sympy.sin,
    SafePower: power_to_sympy,  # Add SafePower to the mapping
    SafeIdentityFunction: sympy.Id  # Add BaseIdentityFunction to the mapping
} 

