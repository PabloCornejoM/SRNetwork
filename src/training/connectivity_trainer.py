import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import random
from concurrent.futures import ProcessPoolExecutor
import copy

class ConnectivityTrainer:
    def __init__(
        self,
        model: 'ConnectivityEQLModel',
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.writer = SummaryWriter(log_dir='logs/connectivity_training')
        
    def train_architecture(
        self,
        architecture: List[torch.Tensor],
        num_parallel_trials: int = 3
    ) -> List[Dict]:
        """Train a single architecture with multiple trials."""
        trial_results = []
        
        for trial in range(num_parallel_trials):
            # Create a fresh copy of the model for each trial
            trial_model = copy.deepcopy(self.model)
            trial_model.build_with_connectivity(architecture)
            
            # Select learning strategy based on trial
            learning_strategy = self._select_learning_strategy(trial)
            
            # Train the model using standard trainer
            from src.training.trainer import Trainer
            trial_trainer = Trainer(
                model=trial_model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                config=self._get_trial_config(learning_strategy),
                device=self.device
            )
            
            trial_trainer.train()
            
            # Evaluate and store results
            val_loss = trial_trainer.validate(epoch=-1)
            trial_results.append({
                'model': trial_model,
                'model_state': trial_model.state_dict(),
                'loss': val_loss,
                'architecture': architecture
            })
            
        return trial_results
    

    def train_all_architectures(
        self,
        max_architectures: Optional[int] = None,
        max_patterns_per_layer: Optional[int] = None,
        num_parallel_trials: int = 3
    ) -> Tuple['ConnectivityEQLModel', float, List[torch.Tensor]]:
        """Train multiple architectures and return the best one."""
        architectures = self.model.get_all_valid_architectures(max_patterns_per_layer)
        
        if max_architectures is not None:
            architectures = random.sample(architectures, min(max_architectures, len(architectures)))
            
        print(f"Training {len(architectures)} different architectures")
        
        best_result = {
            'loss': float('inf'),
            'model_state': None,
            'architecture': None
        }
        
        # Train architectures sequentially since they contain unpicklable objects
        for arch in architectures:
            try:
                trial_results = self.train_architecture(arch, num_parallel_trials)
                best_trial = min(trial_results, key=lambda x: x['loss'])
                
                if best_trial['loss'] < best_result['loss']:
                    best_result = best_trial
                    
            except Exception as e:
                print(f"Architecture training failed: {e}")
                    
        # Load best model state
        #self.model.load_state_dict(best_result['model_state'])
        
        return (
            best_result['model'],
            best_result['model_state'],
            best_result['loss'],
            best_result['architecture']
        )
    
    

    def _select_learning_strategy(self, trial: int) -> str:
        """Select learning rate strategy based on trial number."""
        strategies = ['progressive', 'progressive', 'progressive']
        return strategies[trial % len(strategies)]
        
    def _get_trial_config(self, learning_strategy: str) -> Dict[str, Any]:
        """Create config for specific trial based on learning strategy."""
        config = copy.deepcopy(self.config)
        
        if learning_strategy == 'progressive':
            config['training']['scheduler'] = 'progressive'
        elif learning_strategy == 'cyclic':
            config['training']['scheduler'] = 'cyclic'
        else:
            config['training']['scheduler'] = 'cosine'
            
        return config 