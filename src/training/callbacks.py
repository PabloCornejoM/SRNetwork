from pytorch_lightning.callbacks import Callback
import torch

class SimpleProgressCallback(Callback):
    """Simple callback that prints epoch, train loss, and validation loss."""
    
    def __init__(self):
        super().__init__()
        self.train_loss = float('inf')
        
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        print(f"\nEpoch {epoch + 1}/{trainer.max_epochs}")
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.train_loss = outputs['loss'].item()
            
    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = torch.stack(pl_module.validation_step_outputs).mean()
        print(f"Train Loss: {self.train_loss:.6f}, Val Loss: {val_loss:.6f}") 