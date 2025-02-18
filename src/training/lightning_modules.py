import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import torch.nn.functional as F

class BaseEQLModule(pl.LightningModule):
    """Base Lightning Module for EQL training."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any]
    ):
        super().__init__()
        self.model = model
        self.config = config
        self.equation_history = {}
        self.validation_step_outputs = []
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_idx):
        if len(batch) == 2:
            data, target = batch
        else:
            *data_vars, target = batch
            data = torch.stack(data_vars, dim=1)
            
        output = self(data)
        loss = F.mse_loss(output, target, reduction='sum')
        
        # L1 regularization
        reg_strength = self.config['training']['reg_strength']
        if reg_strength > 0:
            l1_loss = reg_strength * self.model.l1_regularization()
            loss += l1_loss
            self.log('train/l1_loss', l1_loss)
            
        self.log('train/loss', loss)
        return {'loss': loss}
        
    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self(data)
        val_loss = F.mse_loss(output, target, reduction='sum')
        self.log('val/loss', val_loss)
        
        # Store equation at validation intervals
        if self.current_epoch % 100 == 0 or self.current_epoch == self.trainer.max_epochs - 1:
            equation = self.model.get_equation()
            self.equation_history[self.current_epoch] = {
                "equation": equation,
                "loss": val_loss.item()
            }
        
        self.validation_step_outputs.append(val_loss)
        return val_loss
        
    def on_validation_epoch_start(self):
        self.validation_step_outputs = []
        
    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log('val/epoch_loss', epoch_average)
        
    def configure_optimizers(self):
        scheduler_type = self.config['training'].get('scheduler', 'cosine')
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['training']['learning_rate']
        )
        
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=1e-6
            )
        elif scheduler_type == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=0.01,
                max_lr=1.0,
                step_size_up=500,
                mode='triangular'
            )
        elif scheduler_type == 'progressive':
            # For progressive, we'll handle LR updates manually in on_train_epoch_start
            return optimizer
            
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }
        
    def on_train_epoch_start(self):
        # Handle progressive learning rate updates
        if self.config['training'].get('scheduler') == 'progressive':
            """if self.current_epoch == 1000:
                for param_group in self.trainer.optimizers[0].param_groups:
                    param_group['lr'] = 0.1
            elif self.current_epoch == 1300:
                for param_group in self.trainer.optimizers[0].param_groups:
                    param_group['lr'] = 1.0"""
                    
    def on_train_end(self):
        # Print equation history
        for epoch, data in self.equation_history.items():
            print(f"Epoch {epoch}:")
            print(f"Equation: {data['equation']}")
            print(f"Loss: {data['loss']}")
            print()


class ConnectivityEQLModule(BaseEQLModule):
    """Lightning Module for Connectivity-based EQL training."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict[str, Any],
        max_architectures: Optional[int] = None,
        max_patterns_per_layer: Optional[int] = None,
        num_parallel_trials: int = 3
    ):
        super().__init__(model, config)
        self.max_architectures = max_architectures
        self.max_patterns_per_layer = max_patterns_per_layer
        self.num_parallel_trials = num_parallel_trials
        self.best_architecture = None
        self.best_loss = float('inf')
        self.current_architecture = None
        self.current_architecture_idx = 0
        self.current_trial = 0
        
    def setup(self, stage: Optional[str] = None):
        """Setup architectures for training."""
        if stage == 'fit':
            self.architectures = self.model.get_all_valid_architectures(self.max_patterns_per_layer)
            if self.max_architectures is not None:
                import random
                self.architectures = random.sample(
                    self.architectures, 
                    min(self.max_architectures, len(self.architectures))
                )
            print(f"Training {len(self.architectures)} different architectures")
            # Initialize with first architecture
            self.current_architecture = self.architectures[0]
            self.model.build_with_connectivity(self.current_architecture)
            
    def train_dataloader(self):
        return self.trainer.train_dataloader
        
    def val_dataloader(self):
        return self.trainer.val_dataloaders
        
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        
        # Check if we need to move to the next architecture or trial
        if self.current_epoch > 0 and self.current_epoch % self.config['training']['num_epochs'] == 0:
            self.current_trial += 1
            if self.current_trial >= self.num_parallel_trials:
                self.current_trial = 0
                self.current_architecture_idx += 1
                
                if self.current_architecture_idx < len(self.architectures):
                    self.current_architecture = self.architectures[self.current_architecture_idx]
                    print(f"\nStarting training for architecture {self.current_architecture_idx + 1}/{len(self.architectures)}")
                else:
                    print("\nFinished training all architectures")
                    return
                    
            # Update model with current architecture
            self.model.build_with_connectivity(self.current_architecture)
            
    def on_validation_epoch_end(self):
        """Handle architecture selection based on validation performance."""
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.best_architecture = self.current_architecture
            self.best_model_state = self.model.state_dict()
            print(f"\nNew best architecture found! Loss: {avg_loss:.6f}")
            
        super().on_validation_epoch_end()
            
    def on_train_end(self):
        """Load best architecture and model state."""
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nLoaded best architecture with loss: {self.best_loss:.6f}")
        super().on_train_end() 