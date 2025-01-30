# EQL PyTorch Implementation

This repository contains a PyTorch implementation of Equation Learning (EQL) for symbolic regression tasks.

## Project Structure

```
project_root/
│
├── configs/                 # Configuration files
│   ├── config.yaml         # Main configuration
│   └── logging_config.yaml # Logging configuration
│
├── data/                   # Data directory
│   ├── raw/               # Raw data
│   ├── processed/         # Processed data
│   └── external/          # External data sources
│
├── logs/                   # Logging directory
│   ├── tensorboard/       # Tensorboard logs
│   └── training/          # Training logs
│
├── notebooks/             # Jupyter notebooks
│   └── experiments/       # Experiment notebooks
│
├── scripts/               # Utility scripts
│   └── experiments/      # Experiment runners
│
├── src/                   # Main source code
│   ├── data/             # Data processing modules
│   ├── models/           # Model architectures
│   ├── training/         # Training modules
│   ├── evaluation/       # Evaluation modules
│   └── utils/            # Utility functions
│
├── tests/                # Unit tests
│
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

To run an experiment:

```bash
python scripts/experiments/nguyen1.py
```

## Models

The project includes implementations of:
- EQL Model
- Connectivity EQL Model
- Custom activation functions for symbolic regression

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.
