from src.models.eql_model import ConnectivitySRNetwork

def initialize_model(input_size, output_size, num_layers, function_set, nonlinear_info, min_connections_per_neuron=1, exp_n=None):
    """Initialize the ConnectivityEQLModel."""
    return ConnectivitySRNetwork(
        input_size=input_size,
        output_size=output_size,
        num_layers=num_layers,
        function_set=function_set,
        nonlinear_info=nonlinear_info,
        min_connections_per_neuron=min_connections_per_neuron,
        exp_n=exp_n
    )   