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
        self.count = -3

    def init_parameters(self, input_size, output_size):
        """Initialize parameters for EQL layer integration"""
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.zeros(output_size))
        self.sign_params = nn.Parameter(torch.zeros(output_size))
        
        # Initialize exponents between 1 and 6
        nn.init.uniform_(self.weight, 1.0, 6.0)
        self.count += 1
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
            #return f"{sympy.Abs(x)}^{weight}"
            return f"{(x)}^{weight}"
        else:  # odd power behavior
            #return f"sign({x}) * {sympy.Abs(x)}^{weight}"
            return f"{(x)}^{weight}"
    
    def get_parameters_list(self):
        return[], [self.sign_params, self.weight], []

# Update the SYMPY_MAPPING dictionary to include SafePower
def power_to_sympy(x, **kwargs):
    """Convert power function to sympy expression"""
    weight = kwargs.get('weight', 1)
    sign = kwargs.get('sign', 0)
    
    # Create a visual representation of the power function
    if sign > 0.5:  # even power behavior
        #return f"{sympy.Abs(x)}^{weight}"
        return f"{(x)}^{weight}"
    else:  # odd power behavior
        #return f"sign({x}) * {sympy.Abs(x)}^{weight}"
        return f"{(x)}^{weight}"



from typing import List, Tuple

class SafePower2(BaseSafeFunction):
    """
    SafePower function aligned with mvpower.
    Implements a power function with trainable exponents and sign handling.
    """
    start_from = 1  # Class attribute to match mvpower's behavior

    def __init__(self) -> None:
        """
        Initialize the SafePower layer.
        """
        super().__init__("power", "pow")
        self.hardsigmoid = nn.Hardsigmoid()
        self.weight = None  # To be initialized in init_parameters
        self.bias = None
        self.Wsign = None    # To be initialized in init_sign_params
        self.actualsign = None
        self.start_from = SafePower.start_from  # Instance attribute for flexibility

    def init_parameters(self, input_size: int, output_size: int) -> None:
        """
        Initialize parameters for the SafePower layer.

        Parameters:
        - input_size (int): Number of input features.
        - output_size (int): Number of output features.
        """
        # Initialize weight similar to mvpower's weight_init_values
        if output_size > input_size:
            times = (output_size // input_size) + 1
            identity = np.eye(input_size)
            output_matrix = np.concatenate([(self.start_from + i) * identity for i in range(times)], axis=1)
            Wdata = torch.tensor(output_matrix[:, :output_size], dtype=torch.float32)
            self.weight = nn.Parameter(Wdata)
        elif input_size > 1 and output_size == 1:
            Wdata = 2 * torch.rand((output_size, input_size)) - 1  # Initialized between -1 and 1
            self.weight = nn.Parameter(Wdata)
        else:
            Wdata = torch.rand((output_size, input_size))
            self.weight = nn.Parameter(Wdata)

        # Initialize Wsign based on weight parity (assuming integer weights)
        self.init_sign_params()

        self.bias = nn.Parameter(torch.zeros(output_size))

    def init_sign_params(self) -> None:
        """
        Initialize Wsign based on the parity of the weights.
        If weights are floats, a different initialization strategy is used.
        """
        if torch.is_floating_point(self.weight):
            # For floating-point weights, initialize Wsign to favor odd functions initially
            # This mirrors mvpower's initialization logic where Wsign is -1.5 when weight%2 != 0
            W_even = ((self.weight.data % 2) == 0).float()
            Wsign = 1.5 * W_even - 1.5 * (1 - W_even)
            self.Wsign = nn.Parameter(Wsign.unsqueeze(0))  # Adding batch dimension
        else:
            # If weights are integers, use the same logic as mvpower
            W_even = ((self.weight.data % 2) == 0).float()
            Wsign = 1.5 * W_even - 1.5 * (1 - W_even)
            self.Wsign = nn.Parameter(Wsign.unsqueeze(0))  # Adding batch dimension

    def get_param_list(self) -> Tuple[List[nn.Parameter], List[nn.Parameter], List[nn.Parameter]]:
        """
        Categorize parameters into three groups.

        Returns:
        - Tuple containing three lists of parameters.
        """
        return [], [self.Wsign, self.weight], []

    def change_start(self, function_list: List, has_selector: bool, **kwargs) -> None:
        """
        Increment start_from based on the presence of certain functions.

        Parameters:
        - function_list (List): List of functions to check.
        - has_selector (bool): Indicator if a selector is present.
        """
        haslinear = any(isinstance(x, mvlinear) for x in function_list)
        if haslinear or has_selector:
            self.start_from += 1

    def sign_init_values(self, *shape: int) -> None:
        """
        Re-initialize Wsign based on the current weight values.
        This method can be called if weights are updated post-initialization.

        Parameters:
        - shape (int): Shape for Wsign initialization.
        """
        # Replicating mvpower's sign_init_values logic
        if torch.is_floating_point(self.weight):
            W_even = ((self.weight.data % 2) == 0).float()
            Wsign = 1.5 * W_even - 1.5 * (1 - W_even)
            self.Wsign = nn.Parameter(Wsign.unsqueeze(0))
        else:
            W_even = ((self.weight.data % 2) == 0).float()
            Wsign = 1.5 * W_even - 1.5 * (1 - W_even)
            self.Wsign = nn.Parameter(Wsign.unsqueeze(0))

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input by taking the logarithm of its absolute value.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Preprocessed tensor.
        """
        abs_x = torch.clamp(torch.abs(x), min=0.01)
        log_input = torch.log(abs_x)
        return log_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SafePower layer.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (N, D).

        Returns:
        - torch.Tensor: Output tensor after applying the power function.
        """
        # Store the sign of the input
        self.sign = torch.sign(x)
        
        # Preprocess the input
        x_preprocessed = self.preprocess(x)
        
        # Compute the power without sign
        power_no_sign = torch.exp(torch.matmul(x_preprocessed, self.weight.t()))  # Shape: (N, out_dim)
        
        # Apply hardsigmoid to Wsign to get symmetry factor
        simetry = self.hardsigmoid(self.Wsign)  # Shape: (1, out_dim)
        
        # During evaluation, threshold the symmetry factor
        if not self.training:
            simetry = (simetry > 0.5).float()
        
        self.actualsign = simetry
        
        # Reshape for broadcasting
        sign_reshaped = self.sign.unsqueeze(-1)  # Shape: (N, D, 1) if necessary
        # Depending on the shapes, adjust accordingly. Assuming (N, D) and (1, out_dim)
        # We need to ensure that simetry is broadcasted correctly
        # Here, assuming out_dim == D or similar
        
        # Compute the final sign by blending between 1 and sign(x)
        # simetry * 1 + (1 - simetry) * sign(x)
        # Ensure that simetry has the right shape
        simetry_expanded = simetry.expand_as(power_no_sign)  # Shape: (N, out_dim)
        final_sign = simetry_expanded + self.sign * (1 - simetry_expanded)
        
        # Compute the final output
        power_out = final_sign * power_no_sign  # Shape: (N, out_dim)
        
        return power_out

    def acceptRead(self, read_obj, inputs: torch.Tensor) -> str:
        """
        Generate a string representation of the power function.

        Parameters:
        - read_obj: Object that handles the string representation.
        - inputs (torch.Tensor): Input tensor.

        Returns:
        - str: String representation of the power function.
        """
        simetry = self.hardsigmoid(self.Wsign)
        if not self.training:
            simetry = (simetry > 0.5).float()
        self.actualsign = simetry
        return read_obj.powerStr(self.fn_alias, inputs, w=self.weight, Wsign=self.actualsign)

    def clip(self) -> None:
        """
        Clip the weight parameters to a specified range.
        """
        with torch.no_grad():
            self.weight.clamp_(-2.0, 6.0)  # Aligning with mvpower's clipping range


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

