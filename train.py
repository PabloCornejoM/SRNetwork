from utils.wandb_logger import WandBLogger

def train_model():
    # Initialize model and data
    model = EQLModel(...)
    train_loader = ...
    
    # Configure wandb logger
    config = {
        "learning_rate": 0.001,
        "num_epochs": 100,
        "batch_size": 32,
        "model_architecture": str(model),
        "regularization_strength": 1e-3,
        "threshold": 0.1
    }
    
    logger = WandBLogger(
        project_name="eql-experiments",
        config=config,
        run_name="experiment_1",
        notes="Testing new architecture with sin function"
    )
    
    # Log model architecture
    logger.log_model_architecture(model)
    
    # Train model
    train_eql_model(
        model=model,
        train_loader=train_loader,
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        reg_strength=config["regularization_strength"],
        threshold=config["threshold"],
        logger=logger
    )
    
    # Finish logging
    logger.finish() 