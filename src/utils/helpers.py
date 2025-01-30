import yaml
import torch
import logging
import logging.config
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_logging(logging_config_path: str):
    """Setup logging configuration."""
    with open(logging_config_path, 'r') as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 