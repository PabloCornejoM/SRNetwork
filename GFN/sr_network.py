import torch
import torch.nn as nn



class SRNet(nn.Module):
    def __init__(self, input_size, output_size, num_layers, nonlinear_info, functions):
        super(SRNet, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.nonlinear_info = nonlinear_info
        self.functions = functions

        self.layers = self._build_layers(input_size, output_size)

