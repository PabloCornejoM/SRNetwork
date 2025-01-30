import torch
import logging
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.equation_history = {}
        
        self.setup_training()
        
    def setup_training(self):
        """Setup optimizer, scheduler, and other training components."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs'],
            eta_min=1e-6
        )
        
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.writer = SummaryWriter(log_dir='logs/tensorboard')
        
    def train_epoch(self, epoch: int, reg_strength: float, decimal_penalty: float):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            if len(batch_data) == 2:
                data, target = batch_data
            else:
                *data_vars, target = batch_data
                data = torch.stack(data_vars, dim=1)
                
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Base loss
            loss = self.criterion(output, target)
            
            # L1 regularization
            if reg_strength > 0:
                l1_loss = reg_strength * self.model.l1_regularization()
                loss += l1_loss
                
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            # Log metrics
            step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Loss/train', loss.item(), step)
            self.writer.add_scalar('Reg_strength', reg_strength, step)
            if reg_strength > 0:
                self.writer.add_scalar('L1_loss', l1_loss.item(), step)
                
        return total_loss / len(self.train_loader)
        
    def validate(self, epoch: int):
        """Validate the model."""
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(self.val_loader)
        self.writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        if epoch % 100 == 0 or epoch == self.config['training']['num_epochs'] - 1:
            equation = self.model.get_equation()
            self.equation_history[epoch] = {
                "equation": equation,
                "loss": avg_val_loss
            }
            
        return avg_val_loss
        
    def train(self):
        """Main training loop."""
        num_epochs = self.config['training']['num_epochs']
        reg_strength = self.config['training']['reg_strength']
        decimal_penalty = self.config['training'].get('decimal_penalty', 0.01)
        
        print("Starting training with regularization")
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch, reg_strength, decimal_penalty)
            
            if self.val_loader is not None:
                val_loss = self.validate(epoch)
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}")
                
            self.scheduler.step()
            
        # Print equation history
        for epoch, data in self.equation_history.items():
            print(f"Epoch {epoch}:")
            print(f"Equation: {data['equation']}")
            print(f"Loss: {data['loss']}")
            print() 