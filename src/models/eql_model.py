import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import sympy
from src.models.classes import EqlLayer, Connected, MaskedEqlLayer, MaskedConnected
from src.models.custom_functions import SafeIdentityFunction, SafeLog, SafeExp, SYMPY_MAPPING, SafeSin, SafePower


def train_eql_model(model, train_loader, val_loader, num_epochs, learning_rate=0.001,
                    reg_strength=1e-3, threshold=0.1, logger=None, decimal_penalty=0.01):
    """
    Train EQL model using the three-phase schedule from the paper.
    Added decimal complexity penalty parameter.
    
    Arguments:
        model: EQLModel instance
        train_loader: PyTorch DataLoader containing training data
        num_epochs: total number of epochs to train
        learning_rate: learning rate for optimizer
        reg_strength: L1 regularization strength
        threshold: threshold for weight trimming
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, betas=(0.5))
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01) 
    #optimizer = torch.optim.a
    
    #optimizer = torch.optim.Adam(model.get_opt_dict())
    criterion = nn.MSELoss(reduction='sum')
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,  # Total number of epochs
        eta_min=1e-6      # Minimum learning rate
    )
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=1.0, step_size_up=500, mode='triangular')
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2, epochs = num_epochs, steps_per_epoch=len(train_loader))

    early_threshold = 0.1
    SSA = False
    soft_best = False
    #metrics = [MSE(), NRMSE(), R2()]
    #train_model_c(model, train_loader, val_loader, optimizer, criterion, reg_strength, num_epochs, early_threshold, logger, SSA, soft_best)
    #train_model_c(model, 
    #              train_loader, 
    #              val_loader, 
    #              optimizer, 
    #              criterion,
    #              metrics,
    #              reg_function,
    #              num_epochs, 
    #              early_threshold, 
    #              logger, 
    #              SSA, 
    #              soft_best)    
    
    
    # Phase 1: No regularization (T/4)
    #phase1_epochs = int(num_epochs * 0.25)
    #print("Phase 1: Training without regularization")
    #for epoch in range(phase1_epochs):
        #train_epoch(model, train_loader, optimizer, criterion, reg_strength=0.0, epoch=epoch, num_epoch=phase1_epochs, logger=logger)
            
    # Phase 2: With regularization (7T/10)
    phase2_epochs = int(num_epochs * 1)
    print("Phase 2: Training with regularization")
    equation_history = {}
    for epoch in range(phase2_epochs):


        if epoch == 1000:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        #if epoch == 1300:
        #    optimizer = torch.optim.Adam(model.parameters(), lr=1)

        """if epoch == 500:
            optimizer = torch.optim.Adam(model.parameters(), lr=1)

        if epoch == 600:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        

        if epoch == 1000:
            optimizer = torch.optim.Adam(model.parameters(), lr=1)

        if epoch == 1100:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)"""

        """if epoch%100 == 0:
            temperature = epoch / phase2_epochs
            lr = 0.1 + (0.7 - 0.1) * temperature
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            print("lr", lr)"""
        

        train_epoch(model, train_loader, optimizer, criterion, reg_strength=reg_strength, epoch=epoch, num_epoch=phase2_epochs, logger=logger, decimal_penalty=decimal_penalty)
        # Validate and step the scheduler
        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    loss = criterion(output, target)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch + 1}/{phase2_epochs} - Validation Loss: {avg_val_loss:.6f}")
            
            if (epoch) % 100 == 0 or epoch == phase2_epochs - 1:
                    equation = model.get_equation()
                    equation_history[epoch] = {
                    "equation": equation,
                    "loss": avg_val_loss
                }
            # Step the scheduler based on validation loss
            #print(scheduler.get_last_lr())
            #scheduler.step()

    for item in equation_history:
        print(item, equation_history[item]["equation"], equation_history[item]["loss"])               
        print(" ")
    # Phase 3: No regularization, with weight trimming (T/20)
    #phase3_epochs = int(num_epochs * 0.05)
    #print("Phase 3: Training with weight trimming")
    #model.apply_weight_trimming(threshold)
    #for epoch in range(phase3_epochs):
        #train_epoch(model, train_loader, optimizer, criterion, reg_strength=0.0, epoch=epoch, num_epoch=phase3_epochs, logger=logger)

def train_epoch(model, train_loader, optimizer, criterion, reg_strength, epoch=0, num_epoch=0, logger=None, decimal_penalty=0.01):
    """Train for one epoch with improved exploration-exploitation strategy and decimal complexity penalty."""
    model.train()
    step = epoch * len(train_loader)
    total_loss = 0
    
    for batch_idx, batch_data in enumerate(train_loader):
        # Handle multiple input variables
        if len(batch_data) == 2:  # Single input variable case
            data, target = batch_data
        else:  # Multiple input variables case
            *data_vars, target = batch_data
            data = torch.stack(data_vars, dim=1)  # Stack input variables along dimension 1
            
        optimizer.zero_grad()
        output = model(data)
        
        # Base loss
        loss = criterion(output, target)
        
        # L1 regularization
        if reg_strength > 0:
            l1_loss = reg_strength * model.l1_regularization()
            loss += l1_loss
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if logger:
            metrics = {
                "loss": loss.item(),
                "reg_strength": reg_strength if reg_strength > 0 else 0
            }
            if reg_strength > 0:
                metrics["l1_loss"] = l1_loss.item()
            
            logger.log_metrics(metrics, step + batch_idx)
            logger.log_gradients(model, step + batch_idx)
            logger.log_weights(model, step + batch_idx)
            
            if (step + batch_idx) % 10000 == 0:
                equation = model.get_equation()
                logger.log_equation(equation, step + batch_idx)
                
    return total_loss / len(train_loader)





class EQLModel(nn.Module):
    """
    EQL function learning network in PyTorch.

    Arguments:
        input_size: number of input variables to model. Integer.
        output_size: number of variables outputted by model. Integer.
        num_layers: number of layers in model.
        hyp_set: list of PyTorch functions and their sympy equivalents for equation extraction
        nonlinear_info: list of (u,v) tuples for each hidden layer
        name: model name for identification
    """
    def __init__(self, input_size, output_size, num_layers=4, 
                 hyp_set=None, nonlinear_info=None, name='EQL', exp_n=1, functions=None):
        super(EQLModel, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        #self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.name = name
        self.exp_n = exp_n
       
        self.torch_funcs = hyp_set
        # Create corresponding sympy functions
        self.sympy_funcs = []
        for f in hyp_set:
            if isinstance(f, torch.nn.Identity):
                self.sympy_funcs.append(sympy.Id)
            elif f == torch.sin:
                self.sympy_funcs.append(sympy.sin)
            elif f == torch.cos:
                self.sympy_funcs.append(sympy.cos)
            elif isinstance(f, (SafeIdentityFunction, SafeLog, SafeExp, SafeSin, SafePower)):
                self.sympy_funcs.append(SYMPY_MAPPING[f.__class__])
            else:
                raise ValueError(f"Unknown function type: {type(f)}")
        
        # Default nonlinear info if none provided
        self.nonlinear_info = nonlinear_info #or self._get_nonlinear_info(num_layers-1)
        
        # Generate unary functions list
        self.unary_functions = [
            [j % len(self.torch_funcs) for j in range(self.nonlinear_info[i][0])]
            for i in range(num_layers-1)
        ]
        print(self.unary_functions)

        

        if exp_n == 1 or exp_n == 2 or exp_n == 3 or exp_n == 4 or exp_n == 12:
            self.unary_functions = [[6, 6, 6, 6]]
        if exp_n == 5:
            raise ValueError("This is not a valid experiment number yet")
        if exp_n == 6:
            self.unary_functions = [[0, 6], [5, 5]]
        if exp_n == 7:
            self.unary_functions = [[0, 6], [3, 3]]
        if exp_n == 8:
            self.unary_functions = [[6]]
        if exp_n == 9:
            self.unary_functions = [[0, 6], [5, 5]]
        if exp_n == 10:
            raise ValueError("This is not a valid experiment number yet")
        if exp_n == 11:
            raise ValueError("This is not a valid experiment number yet")
        
        if exp_n == 99:
            self.unary_functions = [[0], [5], [3]]

        if functions is not None:            
            self.unary_functions = [[],[],[]]
            for i, item in enumerate(functions):
                if item == "id":
                    self.unary_functions[i] = [0]
                if item == "log":
                    self.unary_functions[i] = [5]
                if item == "sin":
                    self.unary_functions[i] = [3]
            


        print("Were changing the unary functions here")
        #self.unary_functions = [[6, 6, 6]] #n1
        #self.unary_functions = [[6, 6, 6, 6]]
        #self.unary_functions = [[0, 6], [5, 5]] # n6
        #self.unary_functions = [[0, 6], [3, 3]] # n7
        #self.unary_functions = [[6]] # n8
        #self.unary_functions = [[0, 6], [5, 5]] # n9
        
        #self.unary_functions = [[5, 5]] # this is the power function
        #self.unary_functions = [[6, 6, 6, 6]] # this is the power function
        #self.unary_functions = [[3]] # this is the log function
        #self.unary_functions = [[5]] # this is the sin function
        # Build layers
        self.layers = nn.ModuleList()
        inp_size = input_size
        
        # Build hidden layers
        for i in range(num_layers - 1):
            u, v = self.nonlinear_info[i]
            #Change from the last version since here we dont manage this
            #out_size = sum(self.hidden_dim[i]) + 2 * v
            out_size = u + 2 * v
            # Calculate standard deviation for weight initialization
            stddev = np.sqrt(1.0 / (inp_size * out_size))
            #agregar el tema de diemnsion para una f2
            
            
            
            # Add EQL layer
            layer = EqlLayer(
                input_size=inp_size,
                node_info=(u, v),
                #hidden_dim=self.hidden_dim[i],
                hyp_set=self.torch_funcs,  # Only PyTorch functions
                unary_funcs=self.unary_functions[i],
                init_stddev=stddev
            )
            self.layers.append(layer)
            
            # Update input size for next layer
            print(inp_size, out_size)
            inp_size = u + 1 * v
            
            
        # Add final linear layer
        stddev = np.sqrt(1.0 / (inp_size * output_size))
        self.output_layer = Connected(
            input_size=inp_size,
            output_size=output_size,
            init_stddev=stddev
        )

    def _get_nonlinear_info(self, num_hidden_layers, num_binary=[4], unary_per_binary=4):
        """Generate default nonlinear info configuration."""
        nonlinear_info = []
        for _ in range(num_hidden_layers):
            v = np.random.choice(num_binary)  # binary nodes
            u = unary_per_binary * v  # unary nodes
            nonlinear_info.append((u, v))
        return nonlinear_info

    def forward(self, x):
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

    def get_equation(self):
        """Prints learned equation of a model."""
        import sympy as sp
        
        # Create symbolic variables for inputs
        x_symbols = [sp.Symbol(f'x{i+1}') for i in range(self.input_size)]
        X = sp.Matrix([x_symbols])
        
        # Process each layer
        for i, layer in enumerate(self.layers):
            # Get weights and biases with masks applied
            W = layer.W.detach().numpy() * layer.W_mask.detach().numpy()
            #b = layer.b.detach().numpy() * layer.b_mask.detach().numpy()
            try:
                sign_params = layer.sign_params.detach().numpy()
            except:
                sign_params = None

            function_classes = layer.function_classes
            
            # Apply nonlinear transformations
            u, v = layer.node_info
            Y = sp.zeros(u + v, 1)
            
            # Initialize the starting index for Y
            current_index = 0

            # Apply unary functions
            for j in range(u):
                func = function_classes[j]
                if isinstance(func, SafePower):
                    x_term = X[0, 0]  # Use original input
                    if sign_params[current_index] > 0.5:
                        #Y[current_index, 0] = sp.Abs(x_term)**W[current_index]
                        Y[current_index, 0] = (x_term)**W[current_index ]
                    else:
                        #Y[current_index, 0] = sp.sign(x_term) * (sp.Abs(x_term)**W[current_index])
                        Y[current_index, 0] = (x_term)**W[current_index]
                
                elif isinstance(func, SafeIdentityFunction):
                    x_term = X[0, 0]
                    Y[current_index, 0] = x_term
                else:
                    func_idx = layer.unary_funcs[j]
                    x_term = sum(W[current_index, k] * X[k, 0] for k in range(X.rows)) #+ b[current_index]
                    Y[current_index, 0] = self.sympy_funcs[func_idx](x_term)
                current_index += 1
            
            # Apply binary functions
            for j in range(v):
                Y[j + u, 0] = X[u + 2 * j, 0] * X[u + 2 * j + 1, 0]
            
            X = Y
        
        # Final layer
        W = self.output_layer.W.detach().numpy() * self.output_layer.W_mask.detach().numpy()
        #b = self.output_layer.b.detach().numpy() * self.output_layer.b_mask.detach().numpy()
        
        if isinstance(self.torch_funcs[self.unary_functions[0][0]], SafePower):
            W = np.where(np.abs(W) > 1e-5, W, 0)
            #b = np.where(np.abs(b) > 1e-5, b, 0)
        
        W_sp = sp.Matrix(W)
        #b_sp = sp.Matrix(b)
        X = W_sp * X #+ b_sp
        
        # Simplify the expressions
        for i in range(X.cols):
            expr = X[0, i]
            expr = expr.xreplace({n: round(n, 5) for n in expr.atoms(sp.Number) if n.is_real})
            expr = expr.xreplace({t: 0 for t in expr.atoms(sp.Add) if t.is_real and abs((t.evalf())) < 1e-5})
            try:
                X[0, i] = sp.simplify(expr)
            except:
                X[0, i] = expr
        
        # Convert to string for logging
        equation_str = "\n".join([f"y{i+1} = {X[0, i]}" for i in range(X.cols)])
        print("Learned Equation:")
        print(equation_str)
        
        return equation_str  # Return the string representation

    def l1_regularization(self):
        """Calculate L1 regularization loss for all layers."""
        reg_loss = 0.0
        for layer in self.layers:
            reg_loss += layer.l1_regularization()
        reg_loss += self.output_layer.l1_regularization()
        return reg_loss

    def apply_weight_trimming(self, threshold):
        """Apply weight trimming to all layers."""
        for layer in self.layers:
            layer.apply_weight_trimming(threshold)
        self.output_layer.apply_weight_trimming(threshold)

    def __str__(self):
        """Print a structured representation of the model."""
        model_str = f"\nEQL Model: {self.name}\n"
        model_str += "=" * 50 + "\n"
        
        # Input information
        model_str += f"Input size: {self.input_size}\n"
        model_str += f"Output size: {self.output_size}\n"
        model_str += f"Number of layers: {self.num_layers}\n\n"
        
        # Available functions
        model_str += "Activation Functions:\n"
        for i, func in enumerate(self.torch_funcs):
            func_name = func.__class__.__name__ if isinstance(func, torch.nn.Module) else func.__name__
            model_str += f"  [{i}] {func_name}\n"
        model_str += "\n"
        
        # Layer information
        for i, layer in enumerate(self.layers):
            model_str += f"Layer {i+1} (EqlLayer):\n"
            u, v = layer.node_info
            model_str += f"  Unary nodes: {u}\n"
            model_str += f"  Binary nodes: {v}\n"
            model_str += "  Unary functions used:\n"
            for j, func_idx in enumerate(self.unary_functions[i]):
                func = self.torch_funcs[func_idx]
                func_name = func.__class__.__name__ if isinstance(func, torch.nn.Module) else func.__name__
                model_str += f"    Node {j+1}: {func_name}\n"
            model_str += "\n"
        
        # Output layer
        model_str += f"Output Layer:\n"
        model_str += f"  Linear transformation: {self.output_layer.input_size} -> {self.output_layer.output_size}\n"
        
        return model_str

    def decimal_complexity_penalty(self):
        """
        Calculate a penalty based on the decimal complexity of parameters.
        Returns higher values for parameters with more decimal places.
        Excludes sign parameters from SafePower functions.
        """
        penalty = 0.0
        
        def param_decimal_penalty(param, param_name=''):
            # Skip sign parameters from SafePower
            if 'sign_params' in param_name:
                return 0.0
            
            # Convert parameters to float values
            values = param.detach().cpu().numpy().flatten()
            
            # Calculate how far each value is from the nearest integer
            distances = np.abs(values - np.round(values))
            
            # Penalize based on distance to nearest integer
            # Using a smooth function that increases with distance
            penalty = torch.tensor(np.mean(1 - np.exp(-5 * distances)), 
                                 device=param.device, 
                                 dtype=param.dtype)
            return penalty
        
        # Apply to all parameters in the model
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'W' in name or 'b' in name:  # Only apply to weights and biases
                    penalty += param_decimal_penalty(param, name)
        
        # Include output layer parameters
        for name, param in self.output_layer.named_parameters():
            if 'W' in name or 'b' in name:  # Only apply to weights and biases
                penalty += param_decimal_penalty(param, name)
        
        return penalty

class ConnectivityEQLModel(EQLModel):
    """
    Extended EQL model that supports exploring different connectivity patterns between layers.
    """
    def __init__(self, input_size, output_size, num_layers=4,
                 hyp_set=None, nonlinear_info=None, name='ConnectivityEQL',
                 min_connections_per_neuron=1, exp_n=1, functions=None):
        super(ConnectivityEQLModel, self).__init__(
            input_size=input_size,
            output_size=output_size,
            num_layers=num_layers,
            hyp_set=hyp_set,
            nonlinear_info=nonlinear_info,
            exp_n=exp_n, 
            name=name,
            functions=functions
        )
        self.min_connections_per_neuron = min_connections_per_neuron
        
    def generate_valid_patterns(self, m, n, min_connections=1, last=0):
        """
        Generate all valid connection patterns between two layers of sizes m and n.
        
        Args:
            m: Number of neurons in the source layer
            n: Number of neurons in the target layer
            min_connections: Minimum number of connections per neuron
            
        Returns:
            List of valid connectivity matrices (m x n)
        """
        import itertools

        
        
        def is_valid_pattern(matrix):
            # Check source layer connections
            source_connections = [sum(row) for row in matrix]
            if any(conn < min_connections for conn in source_connections):
                return False
                
            # Check target layer connections
            target_connections = [sum(matrix[i][j] for i in range(n)) for j in range(m)]
            if any(conn < min_connections for conn in target_connections):
                return False
            
            # Check SafePower constraint only when it's in the target layer
            for i in range(n):
                # Check if this node uses SafePower function
                if (i < len(self.unary_functions[0]) and  # Only check unary function nodes
                    isinstance(self.torch_funcs[self.unary_functions[0][i]], SafePower)):
                    # Count connections for this node (as target)
                    connections = sum(matrix[i][j] for j in range(m))
                    if connections > 1:
                        return False
                
            return True
        
        patterns = []
        # Generate all possible combinations of connections
        for edges in range(min_connections * max(m, n), m * n + 1):
            # Generate all possible ways to place 'edges' connections
            for combination in itertools.combinations(range(m * n), edges):
                # Create matrix representation
                matrix = [[0] * m for _ in range(n)]
                for idx in combination:
                    i, j = idx // m, idx % m
                    matrix[i][j] = 1
                
                if last == 1:
                    patterns.append(matrix)
                # Check if pattern is valid
                elif is_valid_pattern(matrix):
                    patterns.append(matrix)
        print(patterns)            
        return patterns

    def get_layer_sizes(self):
        """Calculate the size of each layer in the network."""
        layer_sizes = [self.input_size]
        for i in range(self.num_layers - 1):
            u, v = self.nonlinear_info[i]
            layer_sizes.append(u + 2 * v)
        layer_sizes.append(self.output_size)
        return layer_sizes

    def get_all_valid_architectures(self, max_patterns_per_layer=None):
        """
        Generate all valid network architectures based on connectivity constraints.
        
        Args:
            max_patterns_per_layer: Optional int, maximum number of patterns to consider per layer
            
        Returns:
            List of connectivity patterns for each valid architecture
        """
        layer_sizes = self.get_layer_sizes()
        
        # Generate valid patterns for each pair of consecutive layers
        layer_patterns = []
        for i in range(len(layer_sizes) - 1):
            if i == len(layer_sizes) - 2:
                patterns = self.generate_valid_patterns(
                    layer_sizes[i], 
                    layer_sizes[i + 1],
                    self.min_connections_per_neuron,
                    last = 1 
                )

            else:
                patterns = self.generate_valid_patterns(
                    layer_sizes[i], 
                    layer_sizes[i + 1],
                    self.min_connections_per_neuron,
                    last = 0
                )
            
            # Optionally limit the number of patterns per layer
            if max_patterns_per_layer and len(patterns) > max_patterns_per_layer:
                import random
                patterns = random.sample(patterns, max_patterns_per_layer)
                
            layer_patterns.append(patterns)
        
        # Generate all combinations using itertools.product
        import itertools
        all_architectures = list(itertools.product(*layer_patterns))
        
        return all_architectures

    def build_with_connectivity(self, connectivity_patterns):
        """
        Rebuild the model with specified connectivity patterns.
        
        Args:
            connectivity_patterns: List of binary matrices specifying connections between layers
        """
        # Reset layers
        self.layers = nn.ModuleList()
        inp_size = self.input_size
        
        # Build hidden layers
        for i in range(self.num_layers - 1):
            u, v = self.nonlinear_info[i]
            out_size = u + 2 * v
            
            # Calculate standard deviation for weight initialization
            stddev = np.sqrt(1.0 / (inp_size * out_size))
            
            # Add EQL layer with connectivity pattern
            layer = MaskedEqlLayer(
                input_size=inp_size,
                node_info=(u, v),
                #hidden_dim=self.hidden_dim[i],
                hyp_set=self.torch_funcs,
                unary_funcs=self.unary_functions[i],
                init_stddev=stddev,
                connectivity_mask=connectivity_patterns[i]
            )
            self.layers.append(layer)
            
            # Update input size for next layer
            inp_size = u + 1 * v
        
        # Add final linear layer with connectivity pattern
        stddev = np.sqrt(1.0 / (inp_size * self.output_size))
        self.output_layer = MaskedConnected(
            input_size=inp_size,
            output_size=self.output_size,
            connectivity_mask=connectivity_patterns[-1],
            init_stddev=stddev
        )

    def get_trainable_parameters(self):
        """Get all trainable parameters as a flat numpy array."""
        params = []
        for layer in self.layers:
            # Get weights and biases that are actually connected (not masked out)
            if hasattr(layer, 'connectivity_mask'):
                params.extend((layer.W * layer.connectivity_mask).data.flatten().tolist())
            else:
                params.extend(layer.W.data.flatten().tolist())
            #params.extend(layer.b.data.flatten().tolist())
            
            # Get parameters of custom functions if they exist
            if hasattr(layer, 'hyp_set'):
                for func in layer.hyp_set:
                    if hasattr(func, 'parameters'):
                        for param in func.parameters():
                            params.extend(param.data.flatten().tolist())
        
        # Output layer parameters
        if hasattr(self.output_layer, 'connectivity_mask'):
            params.extend((self.output_layer.W * self.output_layer.connectivity_mask).data.flatten().tolist())
        else:
            params.extend(self.output_layer.W.data.flatten().tolist())
        #params.extend(self.output_layer.b.data.flatten().tolist())
        
        return np.array(params)

    def set_trainable_parameters(self, params):
        """Set all trainable parameters from a flat numpy array."""
        with torch.no_grad():
            param_idx = 0
            
            # Set parameters for each layer
            for layer in self.layers:
                # Set weights and biases
                if hasattr(layer, 'connectivity_mask'):
                    mask = layer.connectivity_mask
                    num_weights = int(mask.sum().item())
                    connected_indices = torch.nonzero(mask.flatten()).squeeze()
                    layer.W.data.flatten()[connected_indices] = torch.tensor(
                        params[param_idx:param_idx + num_weights],
                        dtype=torch.float32
                    )
                    param_idx += num_weights
                else:
                    num_weights = layer.W.numel()
                    layer.W.data = torch.tensor(
                        params[param_idx:param_idx + num_weights],
                        dtype=torch.float32
                    ).reshape(layer.W.shape)
                    param_idx += num_weights
                
                #num_biases = layer.b.numel()
                #layer.b.data = torch.tensor(
                #    params[param_idx:param_idx + num_biases],
                #    dtype=torch.float32
                #).reshape(layer.b.shape)
                #param_idx += num_biases
                
                # Set parameters of custom functions if they exist
                if hasattr(layer, 'hyp_set'):
                    for func in layer.hyp_set:
                        if hasattr(func, 'parameters'):
                            for param in func.parameters():
                                num_params = param.numel()
                                param.data = torch.tensor(
                                    params[param_idx:param_idx + num_params],
                                    dtype=torch.float32
                                ).reshape(param.shape)
                                param_idx += num_params
            
            # Set output layer parameters
            if hasattr(self.output_layer, 'connectivity_mask'):
                mask = self.output_layer.connectivity_mask
                num_weights = int(mask.sum().item())
                connected_indices = torch.nonzero(mask.flatten()).squeeze()
                self.output_layer.W.data.flatten()[connected_indices] = torch.tensor(
                    params[param_idx:param_idx + num_weights],
                    dtype=torch.float32
                )
                param_idx += num_weights
            else:
                num_weights = self.output_layer.W.numel()
                self.output_layer.W.data = torch.tensor(
                    params[param_idx:param_idx + num_weights],
                    dtype=torch.float32
                ).reshape(self.output_layer.W.shape)
                param_idx += num_weights
            
            #num_biases = self.output_layer.b.numel()
            #self.output_layer.b.data = torch.tensor(
            #    params[param_idx:param_idx + num_biases],
            #    dtype=torch.float32
            #).reshape(self.output_layer.b.shape)

        

    def optimize_parameters(self, x_data, y_data, method='Nelder-Mead', options=None):
        """
        Optimize model parameters using scipy.optimize.
        
        Args:
            x_data: Input data as numpy array or torch tensor
            y_data: Target data as numpy array or torch tensor
            method: Optimization method for scipy.optimize.minimize
            options: Dictionary of options for the optimizer
            
        Returns:
            OptimizeResult object from scipy.optimize
        """
        from scipy.optimize import minimize
        
        # Convert data to numpy if needed
        if torch.is_tensor(x_data):
            x_data = x_data.detach().numpy()
        if torch.is_tensor(y_data):
            y_data = y_data.detach().numpy()
            
        def loss_function(params):
            """Compute MSE loss for given parameters."""
            self.set_trainable_parameters(params)
            with torch.no_grad():
                x_tensor = torch.tensor(x_data, dtype=torch.float32)
                y_pred = self(x_tensor)
                y_pred = y_pred.detach().numpy()
                return np.mean((y_data - y_pred) ** 2)
        
        # Get initial parameters
        initial_params = self.get_trainable_parameters()
        
        # Set default options if none provided
        if options is None:
            options = {
                'maxiter': 1000,
                'disp': True,
                'adaptive': True
            }
 
        
        # Run optimization
        result = minimize(
            loss_function,
            initial_params,
            method=method,
            options=options
        )
        
        # Update model with best parameters
        self.set_trainable_parameters(result.x)
        
        return result

    def optimize_parameters2(self, x_data, y_data, options=None):
        """
        Optimize model parameters using scipy.optimize.dual_annealing
        
        Args:
            x_data: Input data as numpy array or torch tensor
            y_data: Target data as numpy array or torch tensor
            options: Dictionary of options for the optimizer
            
        Returns:
            OptimizeResult object from scipy.optimize
        """
        from scipy.optimize import dual_annealing
        
        # Convert data to numpy if needed
        if torch.is_tensor(x_data):
            x_data = x_data.detach().numpy()
        if torch.is_tensor(y_data):
            y_data = y_data.detach().numpy()
            
        def loss_function(params):
            """Compute MSE loss for given parameters."""
            self.set_trainable_parameters(params)
            with torch.no_grad():
                x_tensor = torch.tensor(x_data, dtype=torch.float32)
                y_pred = self(x_tensor)
                y_pred = y_pred.detach().numpy()
                return np.mean((y_data - y_pred) ** 2)
        
        # Get initial parameters and set bounds
        initial_params = self.get_trainable_parameters()
        param_bounds = [(-5, 5) for _ in range(len(initial_params))]  # Adjust bounds as needed
        
        # Set default options if none provided
        
        options = {
            'maxiter': 1000,
            'initial_temp': 5230.0,
            'restart_temp_ratio': 2e-5,
            'visit': 2.62,
            'accept': -5.0,
            'maxfun': 10000
        }
        
        # Run optimization
        result = dual_annealing(
            loss_function,
            bounds=param_bounds,
            x0=initial_params,  # Provide initial guess
            seed=42,  # For reproducibility
            **options
        )
        
        # Update model with best parameters
        self.set_trainable_parameters(result.x)
        
        return result

    def train_all_architectures(self, train_loader, val_loader, num_epochs, learning_rate=0.001,
                              reg_strength=1e-3, threshold=0.1, max_architectures=None,
                              max_patterns_per_layer=None, optimize_final=True,
                              optimization_method='Powell', optimization_options=None, 
                              num_parallel_trials=3, logger=None):
        """Train multiple architectures with parallel exploration strategies."""
        architectures = self.get_all_valid_architectures(max_patterns_per_layer)
        if max_architectures is not None:
            import random
            architectures = random.sample(architectures, min(max_architectures, len(architectures)))
            
        best_loss = float('inf')
        best_model = None
        best_architecture = None
        best_optimization_result = None
        
        print(f"Training {len(architectures)} different architectures")
        
        for arch_idx, architecture in enumerate(architectures):
            print(f"\nTraining architecture {arch_idx + 1}/{len(architectures)}")
            
            # Train multiple instances with different strategies
            trial_results = []
            for trial in range(num_parallel_trials):
                # Build new model instance
                self.build_with_connectivity(architecture)
                
                # Different learning rate strategies for each trial
                if trial == 0:
                    # Progressive LR strategy (as implemented in train_epoch)
                    train_strategy = "progressive"
                elif trial == 1:
                    # Cyclic LR strategy
                    optimizer = torch.optim.Adam(self.parameters())
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                    max_lr=1, epochs = num_epochs, steps_per_epoch=len(train_loader))

                else:
                    # Cosine annealing with warm restarts
                    optimizer = torch.optim.Adam(self.parameters())
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizer, T_0=num_epochs//3, T_mult=2, eta_min=0.001
                    )
                
                # Train model
                train_eql_model(
                    self, 
                    train_loader,
                    val_loader,
                    num_epochs,
                    learning_rate,
                    reg_strength,
                    threshold,
                    logger
                )
                
                # Evaluate current model
                self.eval()
                current_loss = self.evaluate_model(val_loader)
                
                # Optimize parameters
                if optimize_final:
                    optimization_result = self.optimize_parameters(
                        *self.get_all_data(train_loader),
                        options=optimization_options
                    )
                    optimized_loss = optimization_result.fun

                else:
                    optimized_loss = current_loss
                    optimization_result = None
                
                trial_results.append({
                    'model_state': self.state_dict(),
                    'loss': optimized_loss,
                    'optimization_result': optimization_result
                })
                print(self.get_equation())
            
            # Select best trial result
            best_trial = min(trial_results, key=lambda x: x['loss'])

            
            if best_trial['loss'] < best_loss:
                best_loss = best_trial['loss']
                best_model = best_trial['model_state']
                best_architecture = architecture
                best_optimization_result = best_trial['optimization_result']
        
        # Load the best model
        self.load_state_dict(best_model)
        
        return self, best_loss, best_architecture, best_optimization_result

    def __str__(self):
        """Print a structured representation of the model with connectivity information."""
        model_str = super(ConnectivityEQLModel, self).__str__()
        
        # Add connectivity information
        model_str += "\nConnectivity Information:\n"
        model_str += "=" * 50 + "\n"
        
        # For each layer
        for i, layer in enumerate(self.layers):
            model_str += f"\nLayer {i+1} Connectivity:\n"
            if hasattr(layer, 'connectivity_mask'):
                mask = layer.connectivity_mask.cpu().numpy()
                model_str += "  Connection matrix:\n"
                for row in mask:
                    model_str += "  " + " ".join(["1" if x else "0" for x in row]) + "\n"
        
        # Output layer connectivity
        if hasattr(self.output_layer, 'connectivity_mask'):
            model_str += "\nOutput Layer Connectivity:\n"
            mask = self.output_layer.connectivity_mask.cpu().numpy()
            model_str += "  Connection matrix:\n"
            for row in mask:
                model_str += "  " + " ".join(["1" if x else "0" for x in row]) + "\n"
        
        return model_str

    def get_opt_dict(self):
        """
        Group parameters by their role and assign different learning rates and beta values.
        Returns a list of parameter dictionaries for the optimizer.
        """
        # Initialize parameter groups
        unary_weights = []  # Weights for unary transformations
        binary_weights = []  # Weights for binary operations
        output_weights = []  # Weights in the output layer
        sign_params = []
        bias_params = []     # All bias parameters

        # Learning rates and betas for different parameter groups
        lr_config = {
            'unary': 0.0001,    # Faster learning for unary transformations
            'binary': 0.005,   # Moderate learning for binary operations
            'sign': 0.005,    # Slower learning for function parameters
            'output': 0.01,   # Moderate-fast learning for output layer
            'bias': 0.005      # Moderate learning for biases
        }

        beta_config = {
            'unary': (0.9, 0.999),
            'binary': (0.9, 0.999),
            'sign': (0.9, 0.999),
            'output': [0.9, 0.999],
            'bias': (0.9, 0.999)
        }

        # Iterate through EQL layers
        for layer in self.layers:
            if isinstance(layer, EqlLayer) or isinstance(layer, MaskedEqlLayer):
                u, v = layer.node_info
                
                # Split weights for unary and binary operations
                unary_weights.extend([
                    param for name, param in layer.named_parameters()
                    if 'W' in name and param.requires_grad
                ][:u])  # First u weights are for unary operations
                
                binary_weights.extend([
                    param for name, param in layer.named_parameters()
                    if 'W' in name and param.requires_grad
                ][u:])  # Remaining weights are for binary operations

                # Collect sign parameters
                sign_params.extend([
                    param for name, param in layer.named_parameters()
                    if 'sign_params' in name and param.requires_grad
                ])
                
                # Collect bias parameters
                bias_params.extend([
                    param for name, param in layer.named_parameters()
                    if 'b' in name and param.requires_grad
                ])

                '''# Collect function-specific parameters
                for func_idx in layer.unary_funcs:
                    func = self.torch_funcs[func_idx]
                    if hasattr(func, 'parameters'):
                        function_params.extend([
                            param for param in func.parameters()
                            if param.requires_grad
                        ])'''

        # Output layer parameters
        if isinstance(self.output_layer, (Connected, MaskedConnected)):
            output_weights.extend([
                param for name, param in self.output_layer.named_parameters()
                if 'W' in name and param.requires_grad
            ])
            bias_params.extend([
                param for name, param in self.output_layer.named_parameters()
                if 'b' in name and param.requires_grad
            ])

        # Create parameter groups with their respective hyperparameters
        param_groups = []
        
        if unary_weights:
            param_groups.append({
                'params': unary_weights,
                'lr': lr_config['unary'],
                'betas': beta_config['unary'],
                'group_name': 'unary'
            })
        
        if binary_weights:
            param_groups.append({
                'params': binary_weights,
                'lr': lr_config['binary'],
                'betas': beta_config['binary'],
                'group_name': 'binary'
            })
        
        if sign_params:  # Sign parameters group
            param_groups.append({
                'params': sign_params,
                'lr': lr_config['sign'],
                'betas': beta_config['sign'],
                'group_name': 'sign'
            })
        
        if output_weights:
            param_groups.append({
                'params': output_weights,
                'lr': lr_config['output'],
                'betas': beta_config['output'],
                'group_name': 'output'
            })
        
        if bias_params:
            param_groups.append({
                'params': bias_params,
                'lr': lr_config['bias'],
                'betas': beta_config['bias'],
                'group_name': 'bias'
            })

        return param_groups

    def evaluate_model(self, val_loader):
        """
        Evaluate the model on a validation dataset.
        
        Args:
            val_loader: PyTorch DataLoader containing validation data
            
        Returns:
            float: Average loss (MSE) on the validation set
        """
        self.eval()  # Set model to evaluation mode
        total_loss = 0.0
        criterion = nn.MSELoss(reduction='sum')
        
        with torch.no_grad():  # Disable gradient computation
            for data, target in val_loader:
                # Forward pass
                output = self(data)
                # Compute loss
                loss = criterion(output, target)
                total_loss += loss.item()
        
        # Compute average loss
        avg_loss = total_loss / len(val_loader.dataset)
        
        return avg_loss

    def get_all_data(self, data_loader):
        """
        Extract all data from a DataLoader and return as input-output pairs.
        
        Args:
            data_loader: PyTorch DataLoader containing the dataset
            
        Returns:
            tuple: (x_data, y_data) where:
                - x_data is a numpy array of input features
                - y_data is a numpy array of target values
        """
        x_list = []
        y_list = []
        
        # Iterate through the DataLoader
        with torch.no_grad():
            for inputs, targets in data_loader:
                # Convert to numpy and store
                x_list.append(inputs.numpy())
                y_list.append(targets.numpy())
        
        # Concatenate all batches
        x_data = np.concatenate(x_list, axis=0)
        y_data = np.concatenate(y_list, axis=0)
        
        return x_data, y_data