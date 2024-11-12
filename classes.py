import torch
import torch.nn as nn
import torch.nn.functional as F

class Connected(nn.Module):
    """
    General-purpose fully connected layer with L1 regularization and weight trimming.

    Arguments:
        input_size: int, number of input features.
        output_size: int, number of output features.
        init_stddev: float, standard deviation for weight initialization.
        regularization: float, L1 regularization coefficient.
    """
    def __init__(self, input_size, output_size, init_stddev=None, regularization=0.0):
        super(Connected, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.regularization = regularization
        self.init_stddev = init_stddev

        # Weight and bias parameters
        self.W = nn.Parameter(torch.Tensor(input_size, output_size))
        self.b = nn.Parameter(torch.Tensor(output_size))

        # Initialize weights and biases
        self.reset_parameters()

        # Masks for weight trimming (non-trainable)
        self.register_buffer('W_mask', torch.ones_like(self.W))
        self.register_buffer('b_mask', torch.ones_like(self.b))

    def reset_parameters(self):
        if self.init_stddev is not None:
            # Use provided standard deviation
            nn.init.normal_(self.W, std=self.init_stddev)
        else:
            # Default to Kaiming initialization
            nn.init.kaiming_normal_(self.W, nonlinearity='linear')
        nn.init.zeros_(self.b)

    def forward(self, x):
        # Apply weight masks for trimming
        W_trimmed = self.W * self.W_mask
        b_trimmed = self.b * self.b_mask
        output = torch.matmul(x, W_trimmed) + b_trimmed
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
    def __init__(self, input_size, node_info, hyp_set, unary_funcs, 
                 init_stddev=None, regularization=0.0):
        u, v = node_info
        output_size = u + 2 * v  # For linear output
        super(EqlLayer, self).__init__(
            input_size, output_size, 
            init_stddev=init_stddev,
            regularization=regularization
        )
        self.node_info = node_info
        self.hyp_set = hyp_set
        self.unary_funcs = unary_funcs

    def forward(self, x):
        # Linear transformation
        lin_output = super(EqlLayer, self).forward(x)

        u, v = self.node_info
        outputs = []

        # Apply unary functions
        for i in range(u):
            func = self.hyp_set[self.unary_funcs[i]]
            outputs.append(func(lin_output[:, i:i+1]))

        # Apply binary functions (products)
        for i in range(u, u + 2 * v, 2):
            prod = lin_output[:, i:i+1] * lin_output[:, i+1:i+2]
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
