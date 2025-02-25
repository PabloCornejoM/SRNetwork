import torch

from src.models.custom_functions import SafeExp, SafeLog, SafeSin, SafePower, SafeIdentityFunction

class Functions:
    def __init__(self):
        self.functions = {
            "exp": SafeExp,
            "log": SafeLog,
            "sin": SafeSin,
            "power": SafePower,
            "identity": SafeIdentityFunction
            # Idea: Add "x" function just to know x in the layer
        }

    def get_function(self, function_name):
        function_class = self.functions.get(function_name)
        if function_class is None:
            raise ValueError(f"Unknown function: {function_name}")
        return function_class()
