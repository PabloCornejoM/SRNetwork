import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.helpers import load_config, setup_logging, set_seed
from src.models.networks import BaseModel
from src.training.trainer import Trainer
from src.data.dataloader import get_dataloaders

def main():
    # Load configurations
    config = load_config('configs/config.yaml')
    setup_logging('configs/logging_config.yaml')
    
    # Set random seed
    set_seed(config['random_seed'])
    
    # Setup data
    train_loader, val_loader = get_dataloaders(config)
    
    # Initialize model
    model = BaseModel(config)
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Train model
    trainer.train()

if __name__ == "__main__":
    main() 