import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect

from src.models.custom_functions import SafePower, SafeIdentityFunction  

class Connected(nn.Module):
    """
    General-purpose fully connected layer with L1 regularization and weight trimming.

    Arguments:
        input_size: int, number of input features.
        output_size: int, number of output features.
        init_stddev: float, standard deviation for weight initialization.
        regularization: float, L1 regularization coefficient.
    """
    def __init__(self, input_size, output_size, init_stddev=None, regularization=0.0, function_classes=None):
        super(Connected, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.regularization = regularization
        self.function_classes = function_classes
        self.init_stddev = init_stddev
        #self.hidden_dim = hidden_dim

        # Weight and bias parameters
        self.W = nn.Parameter(torch.Tensor(output_size, input_size))
        #self.b = nn.Parameter(torch.Tensor(output_size))
        self.sign_params = nn.Parameter(torch.Tensor(output_size))
        # Initialize weights and biases
        self.reset_parameters()

        # Masks for weight trimming (non-trainable)
        self.register_buffer('W_mask', torch.ones_like(self.W))
        #self.register_buffer('b_mask', torch.ones_like(self.b))

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
        if self.function_classes is not None:
            current_index = 0
            # Initialize each segment according to its corresponding function class
            for func_class in self.function_classes:
                #if isinstance(func_class, type):  # Check if it's a class
                    # Create an instance of the function class
                func_instance = func_class
                    # Initialize the parameters using the function class methods
                func_instance.init_parameters(self.input_size, 1)
                    # Copy the initialized parameters to the corresponding segment
                self.W.data[current_index:current_index + 1] = func_instance.weight.data
                #self.b.data[current_index:current_index + 1] = func_instance.bias.data
                try:
                    self.sign_params.data[current_index:current_index + 1] = func_instance.sign_params.data
                except:
                    pass
                current_index += 1
        elif self.init_stddev is not None:
            nn.init.normal_(self.W, std=self.init_stddev)
            #self.W.data.fill_(1)
            #nn.init.uniform_(self.W, a=0.0, b=1.0)  # Initialize W between 0 and 1
            #nn.init.zeros_(self.b)
        else:
            nn.init.kaiming_normal_(self.W, nonlinearity='linear')
            #nn.init.zeros_(self.b)

    def forward(self, x):
        # Apply weight masks for trimming
        W_trimmed = self.W * self.W_mask
        #b_trimmed = self.b * self.b_mask
        #output = torch.matmul(W_trimmed, x.t()) + b_trimmed
        output = torch.matmul(x, W_trimmed.t()) #+ b_trimmed  # Transpose W_trimmed
        return output

    def l1_regularization(self):
        # Calculate L1 regularization term
        reg_loss = self.regularization * (
            torch.sum(torch.abs(self.W * self.W_mask)) #+ 
            #torch.sum(torch.abs(self.b * self.b_mask))
        )
        return reg_loss

    def apply_weight_trimming(self, threshold):
        # Zero out weights and biases below the threshold
        with torch.no_grad():
            self.W_mask.copy_((torch.abs(self.W) >= threshold).float())
            #self.b_mask.copy_((torch.abs(self.b) >= threshold).float())


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
    def __init__(self, input_size, node_info, hyp_set, unary_funcs, 
                 init_stddev=None, regularization=0.0):
        u, v = node_info
        output_size = u + 2 * v
        
        # Get the function classes from hyp_set based on unary_funcs indices
        self.function_classes = []
        for func_idx in unary_funcs:
            self.function_classes.append(hyp_set[func_idx])
            #if inspect.isclass(hyp_set[func_idx]):  # Check if it's a class
                #function_classes.append(hyp_set[func_idx])
            #else:
                #function_classes.append(None)
        
        super(EqlLayer, self).__init__(
            input_size, output_size,
            init_stddev=init_stddev,
            regularization=regularization,
            #hidden_dim=hidden_dim,
            function_classes=self.function_classes
        )
        self.node_info = node_info
        self.hyp_set = hyp_set
        self.unary_funcs = unary_funcs
        #self.hidden_dim = hidden_dim

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
            num_nodes = 1

            if isinstance(func, SafeIdentityFunction):
                segment_output = func(x)  # Pass raw input x instead of transformed
                outputs.append(segment_output)
            
            elif isinstance(func, SafePower):
                # Special handling for SafePower
                #print("self.sign_params", self.sign_params)
                #segment_output = func(x)
                segment_output = func(x, self.W[current_index], self.sign_params[current_index])  # Pass raw input x instead of transformed
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

class MaskedConnected(Connected):
    """
    A variant of Connected layer that supports custom connectivity patterns through masks.
    
    Arguments:
        input_size: int, number of input features
        output_size: int, number of output features
        connectivity_mask: Optional binary matrix specifying allowed connections
        init_stddev: float, standard deviation for weight initialization
        regularization: float, L1 regularization coefficient
    """
    def __init__(self, input_size, output_size, connectivity_mask=None, 
                 init_stddev=None, regularization=0.0, function_classes=None):
        super(MaskedConnected, self).__init__(
            input_size=input_size,
            output_size=output_size,
            init_stddev=init_stddev,
            regularization=regularization,
            function_classes=function_classes
        )
        
        # Initialize connectivity mask
        if connectivity_mask is not None:
            mask = torch.tensor(connectivity_mask, dtype=torch.float32)
            if mask.shape != (output_size, input_size):
                raise ValueError(f"Connectivity mask shape {mask.shape} does not match weight shape {(output_size, input_size)}")
            self.register_buffer('connectivity_mask', mask)
        else:
            self.register_buffer('connectivity_mask', torch.ones(output_size, input_size))
            
        # Apply connectivity mask to weight mask
        self.W_mask.mul_(self.connectivity_mask)
        
    def forward(self, x):
        # Apply both connectivity and trimming masks
        W_masked = self.W * self.W_mask * self.connectivity_mask
        #b_masked = self.b * self.b_mask
        return torch.matmul(x, W_masked.t()) #+ b_masked
        
    def l1_regularization(self):
        # Calculate L1 regularization only for connected weights
        reg_loss = self.regularization * (
            torch.sum(torch.abs(self.W * self.W_mask * self.connectivity_mask)) #+ 
            #torch.sum(torch.abs(self.b * self.b_mask))
        )
        return reg_loss
        
    def apply_weight_trimming(self, threshold):
        # Zero out weights below threshold while respecting connectivity mask
        with torch.no_grad():
            weight_mask = (torch.abs(self.W) >= threshold).float() * self.connectivity_mask
            self.W_mask.copy_(weight_mask)
            self.b_mask.copy_((torch.abs(self.b) >= threshold).float())

class MaskedEqlLayer(EqlLayer):
    """
    A variant of EqlLayer that supports custom connectivity patterns through masks.
    
    Arguments:
        input_size: int, number of input features
        node_info: tuple (u, v), where:
            u: number of unary functions
            v: number of binary functions
        hidden_dim: list of integers specifying number of nodes for each unary function
        hyp_set: list of unary PyTorch functions to be used
        unary_funcs: list of indices specifying which functions from hyp_set to use
        connectivity_mask: Optional binary matrix specifying allowed connections
        init_stddev: float, standard deviation for weight initialization
        regularization: float, L1 regularization coefficient
    """
    def __init__(self, input_size, node_info, hyp_set, unary_funcs,
                 connectivity_mask=None, init_stddev=None, regularization=0.0):
        super(MaskedEqlLayer, self).__init__(
            input_size=input_size,
            node_info=node_info,
            hyp_set=hyp_set,
            unary_funcs=unary_funcs,
            init_stddev=init_stddev,
            regularization=regularization
        )
        
        # Initialize connectivity mask
        u, v = node_info
        output_size = u + 2 * v
        if connectivity_mask is not None:
            mask = torch.tensor(connectivity_mask, dtype=torch.float32)
            if mask.shape != (output_size, input_size):
                raise ValueError(f"Connectivity mask shape {mask.shape} does not match weight shape {(output_size, input_size)}")
            self.register_buffer('connectivity_mask', mask)
        else:
            self.register_buffer('connectivity_mask', torch.ones(output_size, input_size))
            
        # Apply connectivity mask to weight mask
        self.W_mask.mul_(self.connectivity_mask)

    def forward(self, x):
        # Apply connectivity mask to weights before standard forward pass
        self.W_mask.mul_(self.connectivity_mask)
        return super(MaskedEqlLayer, self).forward(x)

    def l1_regularization(self):
        # Include connectivity mask in regularization
        reg_loss = self.regularization * (
            torch.sum(torch.abs(self.W * self.W_mask * self.connectivity_mask)) #+ 
            #torch.sum(torch.abs(self.b * self.b_mask))
        )
        return reg_loss

    def apply_weight_trimming(self, threshold):
        # Apply trimming while respecting connectivity mask
        with torch.no_grad():
            self.W_mask.copy_((torch.abs(self.W) >= threshold).float() * self.connectivity_mask)
            #elf.b_mask.copy_((torch.abs(self.b) >= threshold).float())