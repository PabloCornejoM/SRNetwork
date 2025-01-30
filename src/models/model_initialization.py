from src.models.eql_model import ConnectivityEQLModel

def initialize_model(input_size, output_size, num_layers, hyp_set, nonlinear_info, min_connections_per_neuron=1, exp_n=1):
    """Initialize the ConnectivityEQLModel."""
    return ConnectivityEQLModel(
        input_size=input_size,
        output_size=output_size,
        num_layers=num_layers,
        hyp_set=hyp_set,
        nonlinear_info=nonlinear_info,
        min_connections_per_neuron=min_connections_per_neuron,
        exp_n=exp_n
    )   