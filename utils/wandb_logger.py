import wandb
import torch
import numpy as np
from typing import Optional, Dict, Any

class WandBLogger:
    """Logger class for Weights & Biases integration with EQL models."""
    
    def __init__(self, 
                 project_name: str,
                 config: Optional[Dict[str, Any]] = None,
                 run_name: Optional[str] = None,
                 notes: Optional[str] = None):
        """
        Initialize WandB logger.
        
        Args:
            project_name: Name of the WandB project
            config: Dictionary containing experiment configuration
            run_name: Optional name for this specific run
            notes: Optional notes about this run
        """
        # Initialize wandb
        wandb.init(
            project=project_name,
            config=config,
            name=run_name,
            notes=notes
        )
        
        self.step = 0
        
    def log_gradients(self, model: torch.nn.Module, step: int) -> None:
        """Log gradients of model parameters."""
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[f"gradients/{name}"] = wandb.Histogram(param.grad.cpu().numpy())
                gradients[f"gradient_norm/{name}"] = torch.norm(param.grad).item()
        
        wandb.log(gradients, step=step)
    
    def log_weights(self, model: torch.nn.Module, step: int) -> None:
        """Log model weights."""
        weights = {}
        for name, param in model.named_parameters():
            weights[f"weights/{name}"] = wandb.Histogram(param.data.cpu().numpy())
            weights[f"weight_norm/{name}"] = torch.norm(param.data).item()
            
        wandb.log(weights, step=step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log training metrics."""
        wandb.log(metrics, step=step)
    
    def log_equation(self, equation: str, step: int) -> None:
        """Log the current learned equation."""
        wandb.log({"equation": equation}, step=step)
    
    def log_model_architecture(self, model: torch.nn.Module) -> None:
        """Log model architecture information."""
        wandb.watch(model, log="all")
    
    def log_prediction_plot(self, y_true: np.ndarray, y_pred: np.ndarray, step: int) -> None:
        """Log scatter plot of predictions vs true values."""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, alpha=0.5)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title('Predictions vs True Values')
        
        wandb.log({"predictions": wandb.Image(fig)}, step=step)
        plt.close(fig)
    
    def finish(self) -> None:
        """Finish the logging session."""
        wandb.finish() 