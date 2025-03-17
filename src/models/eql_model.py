import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import sympy
import sympy as sp
import itertools
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
    
    #optimizer = torch.optim.Adam(model.get_selfopt_dict())
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
    SSA = Falseself
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





class SRNetwork(nn.Module):
    """
    SRNetwork function learning network in PyTorch.

    Arguments:
        input_size: number of input variables to model. Integer.
        output_size: number of variables outputted by model. Integer.
        num_layers: number of layers in model.
        fucntion_set: list of PyTorch functions for equation extraction
        nonlinear_info: list of (u,v) tuples for each hidden layer
        name: model name for identification
    """
    
    def __init__(self, input_size, output_size, num_layers=4, 
                 function_set=None, nonlinear_info=None, 
                 name='SRNetwork', exp_n=None):
        super(SRNetwork, self).__init__()
        

        # General parameters
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.name = name
        self.torch_funcs = function_set
        self.nonlinear_info = nonlinear_info
        self.exp_n = exp_n
        self.sympy_funcs = self._create_sympy_functions(function_set)


        if self.exp_n is None:
            # We are in the GFN searching process
            self.structure = self.initialize_funtions_structure()
        
        else:                    
            # When not using GFN, we need to specify the experiment number
            # And the functions to use
            self.structure = self._generate_unary_functions(num_layers - 1)
            self.layers = self._build_layers(input_size, output_size)


    def initialize_funtions_structure(self):
        """Initialize the functions structure for the GFN searching process.
        
        The nonlinear_info format is a list of tuples [(u1, b1), (u2, b2), ...] where:
        - u1, u2 are the number of unary functions for each node
        - b1, b2 are the number of binary functions for each node
        
        For example, if nonlinear_info = [(3, 1), (1, 2)]:
        Returns: [[-1, -1, -1, -2], [-1, -2, -2]] where:
        - -1 represents placeholder for unary functions
        - -2 represents placeholder for binary functions
        """
        structure = []
        
        for unary_count, binary_count in self.nonlinear_info:
            # Create node functions: unary functions (-1) followed by binary functions (-2)
            node_functions = ([-1] * unary_count) + ([-2] * binary_count)
            structure.append(node_functions)
        
        return structure


    
    def _create_sympy_functions(self, function_set):
        """Create corresponding sympy functions from the provided PyTorch functions."""
        sympy_funcs = []
        for f in function_set:
            if f == "identity":
                sympy_funcs.append(sympy.Id)
            elif f == "sin":
                sympy_funcs.append(sympy.sin)
            elif f == "cos":
                sympy_funcs.append(sympy.cos)
            elif f == "log":
                sympy_funcs.append(sympy.log)
            elif f == "exp":
                sympy_funcs.append(sympy.exp)
            elif f == "power":
                sympy_funcs.append(sympy.Pow)
            else:
                raise ValueError(f"Unknown function type: {type(f)}")
        return sympy_funcs
    

    def _generate_unary_functions(self, num_layers):
        """Generate unary functions list based on the experiment number."""
        if self.exp_n in {1, 2, 3, 4, 12}:
            return [["power", "power", "power", "power", "power", "power"]]
        elif self.exp_n == 5:
            raise ValueError("This is not a valid experiment number yet")
        elif self.exp_n == 6:
            return [["identity", "power"], ["log", "log"]]
        elif self.exp_n == 7:
            return [["identity", "power"], ["sin", "sin"]]
        elif self.exp_n == 8:
            return [["power", "power"], ["log", "log"]]
        elif self.exp_n == 9:
            return [["identity", "power"], ["sin", "sin"]]
        elif self.exp_n in {10, 11}:
            raise ValueError("This is not a valid experiment number yet")
        elif self.exp_n == 99:
            return [["sin"]]
        elif self.exp_n == 101:
            return [["sin", "sin"]]
        elif self.exp_n == 71:
            return [["power", "power", "power"], ["log"]]
        elif self.exp_n == 1000:
            return [["sin"]]
        else:
            return [[j % len(self.torch_funcs) for j in range(self.nonlinear_info[i][0])]
                    for i in range(num_layers)]


    def _build_layers(self, input_size, output_size):
        """Build the layers of the model."""
        layers = nn.ModuleList()
        inp_size = input_size
        
        for i in range(self.num_layers - 1):
            u, v = self.nonlinear_info[i]
            out_size = u + 2 * v
            stddev = np.sqrt(1.0 / (inp_size * out_size))
            layer = EqlLayer(
                input_size=inp_size,
                node_info=(u, v),
                function_set=self.torch_funcs,
                unary_funcs=self.structure[i],
                init_stddev=stddev
            )
            layers.append(layer)
            inp_size = u + 1 * v
        
        stddev = np.sqrt(1.0 / (inp_size * output_size))
        self.output_layer = Connected(
            input_size=inp_size,
            output_size=output_size,
            init_stddev=stddev
        )
        return layers

    def forward(self, x):
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

    def get_equation(self):
        """Prints learned equation of a model."""
        import sympy as sp
        x_symbols = [sp.Symbol(f'x{i+1}') for i in range(self.input_size)]
        X = sp.Matrix([x_symbols])
        
        for layer in self.layers:
            X = self._process_layer(X, layer)
        
        W = self.output_layer.W.detach().numpy() * self.output_layer.W_mask.detach().numpy()
        X = sp.Matrix(W) * X
        
        return self._simplify_equation(X)

    def _process_layer(self, X, layer):
        """Process a single layer and return the transformed output."""
        W = layer.W.detach().numpy() * layer.W_mask.detach().numpy()
        sign_params = self._get_sign_params(layer)
        function_classes = layer.function_classes
        u, v = layer.node_info
        Y = sp.zeros(u + v, 1)
        
        current_index = 0
        for j in range(u):
            Y[current_index, 0] = self._apply_unary_function(X, W, function_classes[j], sign_params, current_index)
            current_index += 1
        
        for j in range(v):
            Y[j + u, 0] = X[u + 2 * j, 0] * X[u + 2 * j + 1, 0]
        
        return Y

    def _get_sign_params(self, layer):
        """Retrieve sign parameters from the layer."""
        try:
            return layer.sign_params.detach().numpy()
        except AttributeError:
            return None

    def _apply_unary_function(self, X, W, func, sign_params, current_index):
        """Apply a unary function to the input and return the result."""
        if isinstance(func, SafePower):
            # For power function, we need to handle each input dimension separately
            result = 0
            for j in range(X.shape[1]):
                if abs(W[current_index, j]) > 1e-10:  # Only consider non-zero weights
                    result += (X[0, j]) ** W[current_index, j]
            return result
        
        elif isinstance(func, SafeIdentityFunction):
            return X[0, 0]
        else:
            # Handle other unary functions (log, exp, sin)
            # Get the linear combination of inputs using the weights
            x_term = sum(W[current_index, j] * X[0, j] for j in range(X.shape[1]))
            
            # Get the function name from the function class
            func_name = func.__class__.__name__.lower().replace('safe', '')
            
            # Map the function name to the corresponding sympy function
            if func_name == 'exp':
                return self.sympy_funcs[1](x_term)  # exp
            elif func_name == 'log':
                return self.sympy_funcs[2](x_term)  # log
            elif func_name == 'sin':
                return self.sympy_funcs[3](x_term)  # sin
            else:
                raise ValueError(f"Unknown function type: {func_name}")

    def _simplify_equation(self, X):
        """Simplify the learned equation and return it as a string."""
        for i in range(X.cols):
            expr = X[0, i]
            expr = expr.xreplace({n: round(n, 5) for n in expr.atoms(sp.Number) if n.is_real})
            expr = expr.xreplace({t: 0 for t in expr.atoms(sp.Add) if t.is_real and abs((t.evalf())) < 1e-5})
            X[0, i] = sp.simplify(expr)
        
        return "\n".join([f"y{i+1} = {X[0, i]}" for i in range(X.cols)])

    def l1_regularization(self):
        """Calculate L1 regularization loss for all layers."""
        return sum(layer.l1_regularization() for layer in self.layers) + self.output_layer.l1_regularization()

    def apply_weight_trimming(self, threshold):
        """Apply weight trimming to all layers."""
        for layer in self.layers:
            layer.apply_weight_trimming(threshold)
        self.output_layer.apply_weight_trimming(threshold)

    def __str__(self):
        """Print a structured representation of the model."""
        model_str = f"\nEQL Model: {self.name}\n" + "=" * 50 + "\n"
        model_str += f"Input size: {self.input_size}\nOutput size: {self.output_size}\nNumber of layers: {self.num_layers}\n\n"
        model_str += "Activation Functions:\n" + "\n".join(f"  [{i}] {func.__class__.__name__ if isinstance(func, torch.nn.Module) else func.__name__}" for i, func in enumerate(self.torch_funcs)) + "\n\n"
        
        for i, layer in enumerate(self.layers):
            model_str += f"Layer {i+1} (EqlLayer):\n  Unary nodes: {layer.node_info[0]}\n  Binary nodes: {layer.node_info[1]}\n  Unary functions used:\n"
            model_str += "\n".join(f"    Node {j+1}: {self.torch_funcs[func_idx].__class__.__name__ if isinstance(self.torch_funcs[func_idx], torch.nn.Module) else self.torch_funcs[func_idx].__name__}" for j, func_idx in enumerate(self.structure[i])) + "\n\n"
        
        model_str += f"Output Layer:\n  Linear transformation: {self.output_layer.input_size} -> {self.output_layer.output_size}\n"
        
        return model_str

    def decimal_complexity_penalty(self):
        """Calculate a penalty based on the decimal complexity of parameters."""
        penalty = 0.0
        
        for layer in self.layers:
            penalty += self._calculate_layer_penalty(layer)
        
        penalty += self._calculate_layer_penalty(self.output_layer)
        
        return penalty

    def _calculate_layer_penalty(self, layer):
        """Calculate the penalty for a given layer."""
        return sum(self._param_decimal_penalty(param, name) for name, param in layer.named_parameters() if 'W' in name or 'b' in name)

    def _param_decimal_penalty(self, param, param_name=''):
        """Calculate the decimal complexity penalty for a parameter."""
        if 'sign_params' in param_name:
            return 0.0
        
        values = param.detach().cpu().numpy().flatten()
        distances = np.abs(values - np.round(values))
        return torch.tensor(np.mean(1 - np.exp(-5 * distances)), device=param.device, dtype=param.dtype)


class ConnectivitySRNetwork(SRNetwork):
    """
    Extended SRNetwork model that supports exploring different connectivity patterns between layers.
    """
    def __init__(self, input_size, output_size, num_layers=4,
                 function_set=None, nonlinear_info=None, name='ConnectivitySRNetwork',
                 min_connections_per_neuron=1, exp_n=1):
        super().__init__(input_size=input_size, output_size=output_size,
                         num_layers=num_layers, function_set=function_set,
                         nonlinear_info=nonlinear_info, exp_n=exp_n, 
                         name=name)
        self.min_connections_per_neuron = min_connections_per_neuron

    def generate_valid_patterns(self, source_neurons, target_neurons, min_connections=1, last=0):
        """
        Generate all valid connection patterns between two layers of sizes source_neurons and target_neurons.
        
        Args:
            source_neurons: Number of neurons in the source layer
            target_neurons: Number of neurons in the target layer
            min_connections: Minimum number of connections per neuron
            
        Returns:
            List of valid connectivity matrices (source_neurons x target_neurons)
        """
        import itertools

        def is_valid_pattern(matrix):
            return (self._has_minimum_connections(matrix, min_connections, source_neurons, target_neurons))
        """ and self._check_safe_power_constraint(matrix, target_neurons))"""

        patterns = []
        for edges in range(min_connections * max(source_neurons, target_neurons), source_neurons * target_neurons + 1):
            for combination in itertools.combinations(range(source_neurons * target_neurons), edges):
                matrix = self._create_connection_matrix(combination, source_neurons, target_neurons)
                if last == 1 or is_valid_pattern(matrix):
                    patterns.append(matrix)
        print(patterns)
        return patterns

    def _has_minimum_connections(self, matrix, min_connections, source_neurons, target_neurons):
        source_connections = [sum(row) for row in matrix]
        if any(conn < min_connections for conn in source_connections):
            return False
        target_connections = [sum(matrix[i][j] for i in range(target_neurons)) for j in range(source_neurons)]
        return not any(conn < min_connections for conn in target_connections)

    def _check_safe_power_constraint(self, matrix, target_neurons):
        for i in range(target_neurons):
            if (i < len(self.structure[0]) and
                isinstance(self.torch_funcs[self.structure[0][i]], SafePower)):
                connections = sum(matrix[i][j] for j in range(len(matrix)))
                if connections > 1:
                    return False
        return True

    def _create_connection_matrix(self, combination, source_neurons, target_neurons):
        matrix = [[0] * source_neurons for _ in range(target_neurons)]
        for idx in combination:
            i, j = idx // source_neurons, idx % source_neurons
            matrix[i][j] = 1
        return matrix

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
        layer_patterns = [self._generate_patterns_for_layers(layer_sizes[i], layer_sizes[i + 1], i, max_patterns_per_layer) 
                          for i in range(len(layer_sizes) - 1)]
        return list(itertools.product(*layer_patterns))

    def _generate_patterns_for_layers(self, source_size, target_size, layer_index, max_patterns_per_layer):
        patterns = self.generate_valid_patterns(source_size, target_size, self.min_connections_per_neuron, 
                                                 last=int(layer_index == len(self.nonlinear_info) - 1))
        if max_patterns_per_layer and len(patterns) > max_patterns_per_layer:
            import random
            patterns = random.sample(patterns, max_patterns_per_layer)
        return patterns

    def build_with_connectivity(self, connectivity_patterns):
        """
        Rebuild the model with specified connectivity patterns.
        
        Args:
            connectivity_patterns: List of binary matrices specifying connections between layers
        """
        self.layers = nn.ModuleList()
        inp_size = self.input_size
        
        for i in range(self.num_layers - 1):
            u, v = self.nonlinear_info[i]
            out_size = u + 2 * v
            stddev = np.sqrt(1.0 / (inp_size * out_size))
            layer = MaskedEqlLayer(input_size=inp_size, node_info=(u, v),
                                    function_set=self.torch_funcs, 
                                    unary_funcs=self.structure[i],
                                    init_stddev=stddev, 
                                    connectivity_mask=connectivity_patterns[i])
            self.layers.append(layer)
            inp_size = u + v

        stddev = np.sqrt(1.0 / (inp_size * self.output_size))
        self.output_layer = MaskedConnected(input_size=inp_size, output_size=self.output_size,
                                             connectivity_mask=connectivity_patterns[-1], 
                                             init_stddev=stddev)

    def get_trainable_parameters(self):
        """Get all trainable parameters as a flat numpy array."""
        params = []
        for layer in self.layers:
            params.extend(self._get_layer_parameters(layer))
        params.extend(self._get_layer_parameters(self.output_layer))
        return np.array(params)

    def _get_layer_parameters(self, layer):
        if hasattr(layer, 'connectivity_mask'):
            return (layer.W * layer.connectivity_mask).data.flatten().tolist()
        return layer.W.data.flatten().tolist()

    def set_trainable_parameters(self, params):
        """Set all trainable parameters from a flat numpy array."""
        with torch.no_grad():
            param_idx = 0
            for layer in self.layers:
                param_idx = self._set_layer_parameters(layer, params, param_idx)
            self._set_layer_parameters(self.output_layer, params, param_idx)

    def _set_layer_parameters(self, layer, params, param_idx):
        if hasattr(layer, 'connectivity_mask'):
            mask = layer.connectivity_mask
            num_weights = int(mask.sum().item())
            connected_indices = torch.nonzero(mask.flatten()).squeeze()
            layer.W.data.flatten()[connected_indices] = torch.tensor(
                params[param_idx:param_idx + num_weights], dtype=torch.float32)
            return param_idx + num_weights
        else:
            num_weights = layer.W.numel()
            layer.W.data = torch.tensor(params[param_idx:param_idx + num_weights], dtype=torch.float32).reshape(layer.W.shape)
            return param_idx + num_weights

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
        x_data, y_data = self._convert_data_to_numpy(x_data, y_data)

        def loss_function(params):
            """Compute MSE loss for given parameters."""
            self.set_trainable_parameters(params)
            with torch.no_grad():
                x_tensor = torch.tensor(x_data, dtype=torch.float32)
                y_pred = self(x_tensor).detach().numpy()
                return np.mean((y_data - y_pred) ** 2)

        initial_params = self.get_trainable_parameters()
        options = options or {'maxiter': 1000, 'disp': True, 'adaptive': True}
        result = minimize(loss_function, initial_params, method=method, options=options)
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
        x_data, y_data = self._convert_data_to_numpy(x_data, y_data)

        def loss_function(params):
            """Compute MSE loss for given parameters."""
            self.set_trainable_parameters(params)
            with torch.no_grad():
                x_tensor = torch.tensor(x_data, dtype=torch.float32)
                y_pred = self(x_tensor).detach().numpy()
                return np.mean((y_data - y_pred) ** 2)

        initial_params = self.get_trainable_parameters()
        param_bounds = [(-5, 5) for _ in range(len(initial_params))]
        options = options or {
            'maxiter': 1000,
            'initial_temp': 5230.0,
            'restart_temp_ratio': 2e-5,
            'visit': 2.62,
            'accept': -5.0,
            'maxfun': 10000
        }
        result = dual_annealing(loss_function, bounds=param_bounds, x0=initial_params, seed=42, **options)
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

        best_loss, best_model, best_architecture, best_optimization_result = float('inf'), None, None, None
        print(f"Training {len(architectures)} different architectures")

        for arch_idx, architecture in enumerate(architectures):
            print(f"\nTraining architecture {arch_idx + 1}/{len(architectures)}")
            trial_results = self._train_architecture(architecture, train_loader, val_loader, num_epochs, 
                                                     learning_rate, reg_strength, threshold, 
                                                     optimize_final, optimization_options, 
                                                     num_parallel_trials, logger)

            best_trial = min(trial_results, key=lambda x: x['loss'])
            if best_trial['loss'] < best_loss:
                best_loss, best_model, best_architecture, best_optimization_result = (
                    best_trial['loss'], best_trial['model_state'], architecture, best_trial['optimization_result'])

        self.load_state_dict(best_model)
        return self, best_loss, best_architecture, best_optimization_result

    def _train_architecture(self, architecture, train_loader, val_loader, num_epochs, learning_rate, 
                            reg_strength, threshold, optimize_final, optimization_options, 
                            num_parallel_trials, logger):
        trial_results = []
        for trial in range(num_parallel_trials):
            self.build_with_connectivity(architecture)
            self._train_model(train_loader, val_loader, num_epochs, learning_rate, reg_strength, 
                              threshold, optimize_final, optimization_options, trial_results, trial, logger)
        return trial_results

    def _train_model(self, train_loader, val_loader, num_epochs, learning_rate, reg_strength, 
                     threshold, optimize_final, optimization_options, trial_results, trial, logger):
        train_strategy = self._select_learning_rate_strategy(trial)
        train_eql_model(self, train_loader, val_loader, num_epochs, learning_rate, reg_strength, 
                        threshold, logger)
        self.eval()
        current_loss = self.evaluate_model(val_loader)
        optimized_loss, optimization_result = self._optimize_if_needed(optimize_final, train_loader, 
                                                                      optimization_options, current_loss)
        trial_results.append({
            'model_state': self.state_dict(),
            'loss': optimized_loss,
            'optimization_result': optimization_result
        })
        print(self.get_equation())

    def _select_learning_rate_strategy(self, trial):
        if trial == 0:
            return "progressive"
        elif trial == 1:
            return "cyclic"
        return "cosine"

    def _optimize_if_needed(self, optimize_final, train_loader, optimization_options, current_loss):
        if optimize_final:
            optimization_result = self.optimize_parameters(*self.get_all_data(train_loader), 
                                                           options=optimization_options)
            return optimization_result.fun, optimization_result
        return current_loss, None

    def __str__(self):
        """Print a structured representation of the model with connectivity information."""
        model_str = super().__str__()
        model_str += "\nConnectivity Information:\n" + "=" * 50 + "\n"
        model_str += self._get_connectivity_info()
        return model_str

    def _get_connectivity_info(self):
        connectivity_info = []
        for i, layer in enumerate(self.layers):
            connectivity_info.append(f"\nLayer {i+1} Connectivity:\n")
            if hasattr(layer, 'connectivity_mask'):
                mask = layer.connectivity_mask.cpu().numpy()
                connectivity_info.append("  Connection matrix:\n")
                for row in mask:
                    connectivity_info.append("  " + " ".join(["1" if x else "0" for x in row]) + "\n")
        if hasattr(self.output_layer, 'connectivity_mask'):
            mask = self.output_layer.connectivity_mask.cpu().numpy()
            connectivity_info.append("\nOutput Layer Connectivity:\n")
            connectivity_info.append("  Connection matrix:\n")
            for row in mask:
                connectivity_info.append("  " + " ".join(["1" if x else "0" for x in row]) + "\n")
        return ''.join(connectivity_info)

    def get_opt_dict(self):
        """
        Group parameters by their role and assign different learning rates and beta values.
        Returns a list of parameter dictionaries for the optimizer.
        """
        param_groups = []
        lr_config, beta_config = self._get_learning_configs()

        for layer in self.layers:
            if isinstance(layer, (EqlLayer, MaskedEqlLayer)):
                self._add_layer_params_to_groups(layer, param_groups, lr_config, beta_config)

        if isinstance(self.output_layer, (Connected, MaskedConnected)):
            self._add_layer_params_to_groups(self.output_layer, param_groups, lr_config, beta_config)

        return param_groups

    def _get_learning_configs(self):
        lr_config = {
            'unary': 0.0001,
            'binary': 0.005,
            'sign': 0.005,
            'output': 0.01,
            'bias': 0.005
        }
        beta_config = {
            'unary': (0.9, 0.999),
            'binary': (0.9, 0.999),
            'sign': (0.9, 0.999),
            'output': (0.9, 0.999),
            'bias': (0.9, 0.999)
        }
        return lr_config, beta_config

    def _add_layer_params_to_groups(self, layer, param_groups, lr_config, beta_config):
        unary_weights, binary_weights, sign_params, bias_params = self._collect_layer_params(layer)
        if unary_weights:
            param_groups.append({'params': unary_weights, 'lr': lr_config['unary'], 'betas': beta_config['unary'], 'group_name': 'unary'})
        if binary_weights:
            param_groups.append({'params': binary_weights, 'lr': lr_config['binary'], 'betas': beta_config['binary'], 'group_name': 'binary'})
        if sign_params:
            param_groups.append({'params': sign_params, 'lr': lr_config['sign'], 'betas': beta_config['sign'], 'group_name': 'sign'})
        if bias_params:
            param_groups.append({'params': bias_params, 'lr': lr_config['bias'], 'betas': beta_config['bias'], 'group_name': 'bias'})

    def _collect_layer_params(self, layer):
        unary_weights = [param for name, param in layer.named_parameters() if 'W' in name and param.requires_grad]
        binary_weights = unary_weights[len(unary_weights)//2:]  # Assuming half are unary
        unary_weights = unary_weights[:len(unary_weights)//2]
        sign_params = [param for name, param in layer.named_parameters() if 'sign_params' in name and param.requires_grad]
        bias_params = [param for name, param in layer.named_parameters() if 'b' in name and param.requires_grad]
        return unary_weights, binary_weights, sign_params, bias_params

    def evaluate_model(self, val_loader):
        """
        Evaluate the model on a validation dataset.
        
        Args:
            val_loader: PyTorch DataLoader containing validation data
            
        Returns:
            float: Average loss (MSE) on the validation set
        """
        self.eval()
        total_loss = 0.0
        criterion = nn.MSELoss(reduction='sum')
        
        with torch.no_grad():
            for data, target in val_loader:
                output = self(data)
                loss = criterion(output, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader.dataset)

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
        x_list, y_list = [], []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                x_list.append(inputs.numpy())
                y_list.append(targets.numpy())
        
        return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0)