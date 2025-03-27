import torch


class Log:
    def __init__(self, states, log_probs, rewards, total_flow):
        """
        Initializes a Log object to record sampling statistics from a
        GFlowNet (states, probabilities, and rewards)
        
        Args:
            states: The final sampled states (neural network architectures)
            
            log_probs: The log probabilities of the samples
            
            rewards: The rewards for the sampled neural network architectures
            
            total_flow: The total flow parameter from the GFlowNet
        """
        self._states = states
        self._log_probs = log_probs
        self.rewards = rewards
        self.total_flow = total_flow
    
    @property
    def states(self):
        """
        Returns the final sampled states.
        """
        return self._states
    
    @property
    def fwd_probs(self):
        """
        Returns the forward probabilities (converted from log probabilities).
        """
        return torch.exp(self._log_probs)
    
    @property
    def traj(self):
        """
        Legacy property maintained for compatibility.
        In our neural network implementation, we don't track full trajectories.
        """
        raise NotImplementedError("traj property is not supported in the neural network implementation")
    
    @property
    def actions(self):
        """
        Legacy property maintained for compatibility.
        """
        raise NotImplementedError("actions property is not supported in the neural network implementation")
    
    @property
    def back_probs(self):
        """
        Legacy property maintained for compatibility.
        """
        raise NotImplementedError("back_probs property is not supported in the neural network implementation")
