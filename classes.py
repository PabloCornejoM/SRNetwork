import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

class Connected(nn.Module):
    """
    General-purpose fully connected layer with L1 regularization and weight trimming.

    Arguments:
        input_size: int, number of input features.
        output_size: int, number of output features.
        init_stddev: float, standard deviation for weight initialization.
        regularization: float, L1 regularization coefficient.
    """
    def __init__(self, input_size, output_size, init_stddev=None, regularization=0.0, hidden_dim=None, function_classes=None):
        super(Connected, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.regularization = regularization
        self.function_classes = function_classes
        self.init_stddev = init_stddev
        self.hidden_dim = hidden_dim

        # Weight and bias parameters
        self.W = nn.Parameter(torch.Tensor(output_size, input_size))
        self.b = nn.Parameter(torch.Tensor(output_size))

        # Initialize weights and biases
        self.reset_parameters()

        # Masks for weight trimming (non-trainable)
        self.register_buffer('W_mask', torch.ones_like(self.W))
        self.register_buffer('b_mask', torch.ones_like(self.b))

    def reset_parameters(self):
        """
        Reset the layer's parameters based on initialization settings.

        If function_classes and hidden_dim are provided, initializes segments of weights
        according to the specified function classes. Otherwise, uses either normal
        initialization with specified standard deviation (init_stddev) or Kaiming
        initialization for linear layers.

        The initialization process:
        1. For function classes: Initializes weights/biases per function class specs
        2. For init_stddev: Uses normal distribution with specified std
        3. Default: Uses Kaiming initialization for weights, zeros for biases
        """
        if self.function_classes is not None and self.hidden_dim is not None:
            current_index = 0
            # Initialize each segment according to its corresponding function class
            for func_class, num_nodes in zip(self.function_classes, self.hidden_dim):
                #if isinstance(func_class, type):  # Check if it's a class
                    # Create an instance of the function class
                func_instance = func_class
                    # Initialize the parameters using the function class methods
                func_instance.init_parameters(self.input_size, num_nodes)
                    # Copy the initialized parameters to the corresponding segment
                self.W.data[current_index:current_index + num_nodes] = func_instance.weight.data
                self.b.data[current_index:current_index + num_nodes] = func_instance.bias.data
                current_index += num_nodes
        elif self.init_stddev is not None:
            nn.init.normal_(self.W, std=self.init_stddev)
            nn.init.zeros_(self.b)
        else:
            nn.init.kaiming_normal_(self.W, nonlinearity='linear')
            nn.init.zeros_(self.b)

    def forward(self, x):
        # Apply weight masks for trimming
        W_trimmed = self.W * self.W_mask
        b_trimmed = self.b * self.b_mask
        #output = torch.matmul(W_trimmed, x.t()) + b_trimmed
        output = torch.matmul(x, W_trimmed.t()) + b_trimmed  # Transpose W_trimmed
        return output

    def l1_regularization(self):
        # Calculate L1 regularization term
        reg_loss = self.regularization * (
            torch.sum(torch.abs(self.W * self.W_mask)) + 
            torch.sum(torch.abs(self.b * self.b_mask))
        )
        return reg_loss

    def apply_weight_trimming(self, threshold):
        # Zero out weights and biases below the threshold
        with torch.no_grad():
            self.W_mask.copy_((torch.abs(self.W) >= threshold).float())
            self.b_mask.copy_((torch.abs(self.b) >= threshold).float())


class EqlLayer(Connected):
    """
    EQL linear-nonlinear layer with unary and binary nonlinearities.

    Arguments:
        input_size: int, number of input features
        node_info: tuple (u, v), where:
            u: number of unary functions
            v: number of binary functions
        hyp_set: list of unary PyTorch functions to be used
        unary_funcs: list of indices specifying which functions from hyp_set to use
        init_stddev: float, standard deviation for weight initialization
        regularization: float, L1 regularization coefficient
    """
    def __init__(self, input_size, node_info, hidden_dim, hyp_set, unary_funcs, 
                 init_stddev=None, regularization=0.0):
        u, v = node_info
        output_size = sum(hidden_dim) + 2 * v
        
        # Get the function classes from hyp_set based on unary_funcs indices
        function_classes = []
        for func_idx in unary_funcs:
            function_classes.append(hyp_set[func_idx])
            #if inspect.isclass(hyp_set[func_idx]):  # Check if it's a class
                #function_classes.append(hyp_set[func_idx])
            #else:
                #function_classes.append(None)
        
        super(EqlLayer, self).__init__(
            input_size, output_size,
            init_stddev=init_stddev,
            regularization=regularization,
            hidden_dim=hidden_dim,
            function_classes=function_classes
        )
        self.node_info = node_info
        self.hyp_set = hyp_set
        self.unary_funcs = unary_funcs
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # Linear transformation for non-power functions
        lin_output = super(EqlLayer, self).forward(x)
        
        u, v = self.node_info
        outputs = []
        
        # Process each function
        lin_output_t = lin_output.t()
        current_index = 0
        
        for i in range(u):
            func = self.hyp_set[self.unary_funcs[i]]
            num_nodes = self.hidden_dim[i]
            
            if isinstance(func, SafePower):
                # Special handling for SafePower
                segment_output = func(x)  # Pass raw input x instead of transformed
                outputs.append(segment_output)
            else:
                # Regular handling for other functions
                outputs.append(func(lin_output_t[current_index:current_index + num_nodes]).t())
            
            current_index += num_nodes

        # Apply binary functions (products) row-wise
        for i in range(u, u + 2 * v, 2):
            prod = lin_output_t[i:i+1].t() * lin_output_t[i+1:i+2].t()
            outputs.append(prod)

        # Concatenate outputs
        output = torch.cat(outputs, dim=1)
        return output


class DivLayer(Connected):
    """
    EQL division layer for EQL-div models.

    Arguments:
        input_size: int, number of input features
        output_size: int, number of output features
        threshold: float, minimum denominator value before output is zeroed
        init_stddev: float, standard deviation for weight initialization
        regularization: float, L1 regularization coefficient
    """
    def __init__(self, input_size, output_size, threshold=0.001,
                 init_stddev=None, regularization=0.0):
        super(DivLayer, self).__init__(
            input_size, output_size * 2,  # Double outputs for numerator/denominator pairs
            init_stddev=init_stddev,
            regularization=regularization
        )
        self.true_output_size = output_size
        self.register_buffer('threshold', torch.tensor(threshold))

    def forward(self, x):
        # Linear transformation
        lin_output = super(DivLayer, self).forward(x)
        
        # Split into numerators and denominators
        numerators = lin_output[:, ::2]
        denominators = lin_output[:, 1::2]
        
        # Zero outputs where denominator is below threshold
        zeros = (denominators > self.threshold).float()
        denominators_inverse = 1.0 / (torch.abs(denominators) + 1e-10)
        output = numerators * denominators_inverse * zeros
        
        # Add denominator penalty to regularization
        denominator_loss = torch.sum(
            torch.maximum(self.threshold - denominators, 
                        torch.zeros_like(denominators))
        )
        
        return output
