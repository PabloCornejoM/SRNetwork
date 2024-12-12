import torch
import torch.nn as nn
import torch.nn.functional as F

from classes import EqlLayer, Connected, MaskedEqlLayer, MaskedConnected

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

import sympy
from custom_functions import SafeIdentityFunction, SafeLog, SafeExp, SYMPY_MAPPING, SafeSin, SafePower

def train_eql_model(model, train_loader, num_epochs, learning_rate=0.001,
                    reg_strength=1e-3, threshold=0.1):
    """
    Train EQL model using the three-phase schedule from the paper.
    
    Arguments:
        model: EQLModel instance
        train_loader: PyTorch DataLoader containing training data
        num_epochs: total number of epochs to train
        learning_rate: learning rate for optimizer
        reg_strength: L1 regularization strength
        threshold: threshold for weight trimming
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Phase 1: No regularization (T/4)
    phase1_epochs = int(num_epochs * 0.25)
    print("Phase 1: Training without regularization")
    for epoch in range(phase1_epochs):
        train_epoch(model, train_loader, optimizer, criterion, reg_strength=0.0)
            
    # Phase 2: With regularization (7T/10)
    phase2_epochs = int(num_epochs * 0.7)
    print("Phase 2: Training with regularization")
    for epoch in range(phase2_epochs):
        train_epoch(model, train_loader, optimizer, criterion, reg_strength=reg_strength)
            
    # Phase 3: No regularization, with weight trimming (T/20)
    phase3_epochs = int(num_epochs * 0.05)
    print("Phase 3: Training with weight trimming")
    model.apply_weight_trimming(threshold)
    for epoch in range(phase3_epochs):
        train_epoch(model, train_loader, optimizer, criterion, reg_strength=0.0)

def train_epoch(model, train_loader, optimizer, criterion, reg_strength):
    """Train for one epoch."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Add L1 regularization
        if reg_strength > 0:
            loss += reg_strength * model.l1_regularization()
            
        loss.backward()
        optimizer.step()




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
                 hyp_set=None, nonlinear_info=None, name='EQL'):
        super(EQLModel, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        #self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.name = name

       
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

        #self.unary_functions = [[0, 6], [5, 5]] # which unary functions to use per layer
        self.unary_functions = [[6]] # this is the power function
        self.unary_functions = [[3]] # this is the log function

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
        """Prints learned equation of a trained model."""
        import sympy as sp
        
        # Create symbolic variables for inputs
        x_symbols = [sp.Symbol(f'x{i+1}') for i in range(self.input_size)]
        X = sp.Matrix([x_symbols])
        
        # Process each layer
        for i, layer in enumerate(self.layers):
            # Get weights and biases with masks applied
            W = layer.W.detach().numpy() * layer.W_mask.detach().numpy()
            b = layer.b.detach().numpy() * layer.b_mask.detach().numpy()
            
            # Linear transformation
            W_sp = sp.Matrix(W)
            b_sp = sp.Matrix(b)
            X = W_sp * X + b_sp
            
            # Apply nonlinear transformations
            u, v = layer.node_info
            Y = sp.zeros(u + v, 1)
            
            # Initialize the starting index for Y
            current_index = 0

            # Apply unary functions
            for j in range(u):
                func_idx = layer.unary_funcs[j]
                # Loop through the number of nodes for the current unary function
                for k in range(1):
                    Y[current_index, 0] = self.sympy_funcs[func_idx](X[current_index, 0])  # Use sympy function
                    current_index += 1  # Move to the next index in Y
            
            # Apply binary functions (products)
            for j in range(v):
                Y[j + u, 0] = X[u + 2 * j, 0] * X[u + 2 * j + 1, 0]  # Ensure correct access to X
            
            X = Y
        
        # Final layer
        W = self.output_layer.W.detach().numpy() * self.output_layer.W_mask.detach().numpy()
        b = self.output_layer.b.detach().numpy() * self.output_layer.b_mask.detach().numpy()
        W_sp = sp.Matrix(W)
        b_sp = sp.Matrix(b)
        X = W_sp * X + b_sp
        
        print("Learned Equation:")
        for i in range(X.cols):
            print(f"y{i+1} = {X[0, i]}")
        
        return X

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

class ConnectivityEQLModel(EQLModel):
    """
    Extended EQL model that supports exploring different connectivity patterns between layers.
    """
    def __init__(self, input_size, output_size, num_layers=4,
                 hyp_set=None, nonlinear_info=None, name='ConnectivityEQL',
                 min_connections_per_neuron=1):
        super(ConnectivityEQLModel, self).__init__(
            input_size=input_size,
            output_size=output_size,
            #hidden_dim=hidden_dim,
            num_layers=num_layers,
            hyp_set=hyp_set,
            nonlinear_info=nonlinear_info,
            name=name
        )
        self.min_connections_per_neuron = min_connections_per_neuron
        
    def generate_valid_patterns(self, m, n, min_connections=1):
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
                    
                # Check if pattern is valid
                if is_valid_pattern(matrix):
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
            patterns = self.generate_valid_patterns(
                layer_sizes[i], 
                layer_sizes[i + 1],
                self.min_connections_per_neuron
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

    def train_all_architectures(self, train_loader, num_epochs, learning_rate=0.001,
                              reg_strength=1e-3, threshold=0.1, max_architectures=None,
                              max_patterns_per_layer=None):
        """
        Train all valid architectures and return the best performing one.
        
        Args:
            train_loader: PyTorch DataLoader containing training data
            num_epochs: Number of epochs to train each architecture
            learning_rate: Learning rate for optimization
            reg_strength: L1 regularization strength
            threshold: Threshold for weight trimming
            max_architectures: Maximum number of architectures to try
            max_patterns_per_layer: Maximum number of patterns to consider per layer
            
        Returns:
            best_model: The best performing model
            best_loss: The loss of the best model
            best_architecture: The connectivity pattern of the best model
        """
        architectures = self.get_all_valid_architectures(max_patterns_per_layer)
        if max_architectures is not None:
            import random
            architectures = random.sample(architectures, min(max_architectures, len(architectures)))
            
        best_loss = float('inf')
        best_model = None
        best_architecture = None
        
        print(f"Training {len(architectures)} different architectures")
        
        for arch_idx, architecture in enumerate(architectures):
            print(f"\nTraining architecture {arch_idx + 1}/{len(architectures)}")
            
            # Build model with current architecture
            self.build_with_connectivity(architecture)
            
            # Train the model
            train_eql_model(
                self, 
                train_loader, 
                num_epochs, 
                learning_rate,
                reg_strength, 
                threshold
            )
            
            # Evaluate the model
            self.eval()
            total_loss = 0
            criterion = nn.MSELoss()
            with torch.no_grad():
                for data, target in train_loader:
                    output = self(data)
                    loss = criterion(output, target)
                    total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Architecture {arch_idx + 1} - Average Loss: {avg_loss:.6f}")

            print(self.get_equation())
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = self.state_dict()
                best_architecture = architecture
                
        # Load the best model
        self.load_state_dict(best_model)
        return self, best_loss, best_architecture
        
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