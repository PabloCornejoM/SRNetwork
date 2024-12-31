import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import datetime
import numpy as np

class TensorBoardLogger:
    def __init__(self, log_dir="runs"):
        # Create a unique run name with timestamp
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = Path(log_dir) / current_time
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
    def log_loss(self, loss, step, prefix='train'):
        """Log loss values"""
        self.writer.add_scalar(f'{prefix}/loss', loss, step)
    
    def log_gradients(self, model, step):
        """Log model gradients"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, step)
    
    def log_weights(self, model, step):
        """Log model weights"""
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'weights/{name}', param.data, step)
    
    def log_architecture(self, model, step):
        """Log model architecture information"""
        self.writer.add_text('architecture/structure', str(model), step)
        if hasattr(model, 'get_equation'):
            self.writer.add_text('architecture/equation', model.get_equation(), step)
    
    def log_prediction_plot(self, x_values, y_true, y_pred, step):
        """Log prediction vs true value plot"""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_values, y_true, label='True')
        ax.plot(x_values, y_pred, '--', label='Prediction')
        ax.set_title('Predictions vs True Values')
        ax.legend()
        ax.grid(True)
        self.writer.add_figure('predictions', fig, step)
        plt.close(fig)
        
    
    def close(self):
        """Close the writer"""
        self.writer.close() 