import pytorch_lightning as pl
from typing import Dict, Any
from src.training.lightning_modules import BaseSRNetModule, ConnectivitySRNetModule
from src.training.callbacks import SimpleProgressCallback

# This is the main training script for the SRNet model when using Lightning
# It is used to train the model using the Lightning framework, if not it will use the original trainer


def train_model(model, train_loader, val_loader, config, device):
    """
    Train an SRNet model using PyTorch Lightning.
    
    Args:
        model: The SRNet model to train
        train_loader: PyTorch DataLoader for training data
        val_loader: PyTorch DataLoader for validation data
        config: Dictionary containing training configuration
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        The trained model
    """
    
    trainer_config = {
        'max_epochs': config['training']['num_epochs'],
        'accelerator': 'gpu' if device == 'cuda' else 'cpu',
        'devices': 1,
        'logger': pl.loggers.TensorBoardLogger('logs/lightning_logs'),
        'enable_checkpointing': True,
        'enable_progress_bar': False,  # Disable default progress bar
        
        # Precision settings # Create Lightning trainer with specified configuration
        'precision': config['training'].get('precision', '32'),  # Default to 32-bit,  
        
        # For 8-bit training (if using precision="8")
        'plugins': [pl.plugins.precision.BitsandbytesPrecision(mode="bf16-mixed")] if precision == "8" else None,
        
        'callbacks': [
            SimpleProgressCallback(),  # Our custom callback
            pl.callbacks.ModelCheckpoint(
                monitor='val/loss',
                mode='min',
                save_top_k=1,
                filename='best-{epoch:02d}-{val/loss:.6f}'
            ),
            pl.callbacks.EarlyStopping(
                monitor='val/loss',
                patience=50,
                mode='min'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ]
    }
    
    # Remove None values from config
    trainer_config = {k: v for k, v in trainer_config.items() if v is not None}
    
    trainer = pl.Trainer(**trainer_config)
    
    # Create appropriate Lightning module based on training type
    if config['training'].get('use_connectivity_training', False):
        lightning_module = ConnectivitySRNetModule(
            model=model,
            config=config,
            max_architectures=config['training'].get('max_architectures'),
            max_patterns_per_layer=config['training'].get('max_patterns_per_layer'),
            num_parallel_trials=config['training'].get('num_parallel_trials', 3)
        )
    else:
        lightning_module = BaseSRNetModule(
            model=model,
            config=config
        )
    
    # Train the model
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    return lightning_module.model