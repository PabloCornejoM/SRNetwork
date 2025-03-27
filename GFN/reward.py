import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any
from src.training.trainer import Trainer

class RewardManager:
    """Base class for reward management in GFlowNet training."""
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.best_reward = float('-inf')
        self.best_model_state = None
        self.best_architecture = None

    def _update_rewards(self, rewards: torch.Tensor, loss: torch.Tensor, model_state: Dict[str, Any], architecture: list):
        """Update best reward and model state if current reward is better."""
        if len(rewards) > 1 and torch.max(rewards) > self.best_reward:
            best_reward = torch.max(rewards)
            best_idx = torch.argmax(rewards)
            if self.verbose:
                print(f"\nNew best reward: {best_reward}")
                print(f"Loss: {loss[best_idx]}")
            self.best_reward = best_reward
            self.best_model_state = model_state
            self.best_architecture = architecture

    def calc_rewards(self, loss: torch.Tensor, model_state: Dict[str, Any], architecture: list, is_eval: bool = False):
        """Calculate rewards based on loss and model state."""
        raise NotImplementedError("Cannot use an abstract reward manager class")

class NRMSEReward(RewardManager):
    """Reward based on Normalized Root Mean Square Error."""
    def __init__(self, y_std: float, loss_threshold: float = 0.1, verbose: bool = True):
        super().__init__(verbose)
        self.y_std = y_std
        self.loss_threshold = loss_threshold

    def calc_rewards(self, loss: torch.Tensor, model_state: Dict[str, Any], architecture: list, is_eval: bool = False):
        nrmse = torch.sqrt(loss) / self.y_std
        rewards = torch.clamp(self.loss_threshold / (self.loss_threshold + nrmse), min=0.01)
        if not is_eval:
            self._update_rewards(rewards, loss, model_state, architecture)
        return rewards

class TSSReward(RewardManager):
    """Reward based on Total Sum of Squares."""
    def __init__(self, max_mse: float, verbose: bool = True):
        super().__init__(verbose)
        self.max_mse = max_mse

    def calc_rewards(self, loss: torch.Tensor, model_state: Dict[str, Any], architecture: list, is_eval: bool = False):
        rewards = torch.clamp(1.0 - loss / self.max_mse, min=0.01)
        if not is_eval:
            self._update_rewards(rewards, loss, model_state, architecture)
        return rewards

class DynamicTSSReward(RewardManager):
    """Reward based on Dynamic Total Sum of Squares with adaptive baseline."""
    def __init__(self, initial_baseline_mse: float, gamma: float = 0.1, q: float = 0.9, verbose: bool = True):
        super().__init__(verbose)
        self.baseline_mse = initial_baseline_mse
        self.gamma = gamma
        self.q = q

    def calc_rewards(self, loss: torch.Tensor, model_state: Dict[str, Any], architecture: list, is_eval: bool = False):
        rewards = torch.clamp(1.0 - loss / self.baseline_mse, min=1e-4)
        
        if not is_eval:
            self._update_rewards(rewards, loss, model_state, architecture)
            
            # Update baseline MSE
            new_baseline_mse = self.gamma * self.baseline_mse + (1 - self.gamma) * torch.min(loss)
            if new_baseline_mse < self.baseline_mse:
                self.baseline_mse = new_baseline_mse
                
        return rewards

class StructureReward(RewardManager):
    """Reward based on model structure complexity."""
    def __init__(self, verbose: bool = True):
        super().__init__(verbose)

    def calc_rewards(self, loss: torch.Tensor, model_state: Dict[str, Any], architecture: list, is_eval: bool = False):
        # Count number of non-zero parameters
        num_params = sum(p.numel() for p in model_state.values() if p.requires_grad)
        rewards = torch.ones(len(loss))
        rewards[num_params > 1000] = 1e-4  # Penalize complex architectures
        
        if not is_eval:
            self._update_rewards(rewards, loss, model_state, architecture)
        return rewards

def compute_reward(model, train_loader, val_loader, config, device, reward_type='nrmse'):
    """
    Compute reward for a given model architecture.
    
    Args:
        model: The trained model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Device to use for computation
        reward_type: Type of reward to compute ('nrmse', 'tss', 'dynamic', 'struct')
        
    Returns:
        torch.Tensor: Rewards for the model
    """
    # Train the model
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    trainer.train()
    
    # Get validation loss
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.mse_loss(output, target, reduction='sum')
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader.dataset)
    
    # Initialize appropriate reward manager
    if reward_type == 'nrmse':
        y_std = torch.std(torch.cat([target for _, target in val_loader])).item()
        reward_manager = NRMSEReward(y_std=y_std)
    elif reward_type == 'tss':
        max_mse = ((torch.cat([target for _, target in val_loader]) - 
                   torch.cat([target for _, target in val_loader]).mean()) ** 2).mean().item()
        reward_manager = TSSReward(max_mse=max_mse)
    elif reward_type == 'dynamic':
        initial_baseline = ((torch.cat([target for _, target in val_loader]) - 
                           torch.cat([target for _, target in val_loader]).mean()) ** 2).mean().item()
        reward_manager = DynamicTSSReward(initial_baseline_mse=initial_baseline)
    elif reward_type == 'struct':
        reward_manager = StructureReward()
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
    
    # Calculate rewards
    loss_tensor = torch.tensor([avg_loss])
    rewards = reward_manager.calc_rewards(
        loss=loss_tensor,
        model_state=model.state_dict(),
        architecture=model.structure,
        is_eval=False
    )
    
    return rewards 