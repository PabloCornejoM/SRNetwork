import torch
import numpy as np
import math
from copy import deepcopy

def train_model_c(model, 
               train_loader, 
               val_loader, 
               optimizer_config, 
               criterion, 
               metrics, 
               reg_function, 
               num_epochs, 
               early_threshold, 
               logger=None,
               SSA=False,
               soft_best=False):
    """
    Comprehensive training function incorporating multiple loss components, validation, early stopping,
    model selection, and post-step operations.
    
    Parameters:
        model (nn.Module): The symbolic regression model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer_config (list of dict): Configuration for optimizer parameter groups.
        criterion (nn.Module): Loss function (e.g., MSELoss).
        metrics (list of functions): List of metric functions to evaluate.
        reg_function (function): Regularization function.
        num_epochs (int): Total number of epochs to train.
        early_threshold (float): Fraction of total epochs to determine early stopping patience.
        logger (Logger, optional): Logger for tracking metrics.
        SSA (bool): Flag for selecting validation metric.
        soft_best (bool): Flag for soft model selection.
    
    Returns:
        dict: Training results including best model, loss histories, and other metrics.
    """
    # Initialize optimizer with multiple parameter groups
    #optimizer = torch.optim.Adam(optimizer_config)
    optimizer = optimizer_config

    
    # Initialize loss histories
    train_losses = []
    reg_losses = []
    complex_losses = []
    val_eq_losses = []
    val_nn_losses = []
    val_eq_loss_orig = []
    val_eq_loss_complex = []
    
    # Regularization setup
    reg = reg_function
    
    # Early stopping variables
    early_stop_patience = math.ceil(early_threshold * num_epochs)
    early_counter = 0
    best_model = None
    best_value = None
    best_param_n = None
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_losses = []
        epoch_reg_losses = []
        epoch_complex_losses = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data, target = data.float(), target.float()
            output = model(data)
            target_loss = criterion(output, target)
            reg_loss = reg(model)
            complexity_loss = len(model.to_string())
            
            total_loss = target_loss + reg_loss + 0 * complexity_loss  # Modify if complexity is to be included differently
            
            total_loss.backward()
            optimizer.step()
            model.post_step()
            
            # Collect losses
            epoch_train_losses.append(target_loss.item())
            epoch_reg_losses.append(reg_loss.item())
            epoch_complex_losses.append(complexity_loss)
            
            # Logging
            if logger:
                metrics_log = {
                    "loss": target_loss.item(),
                    "reg_loss": reg_loss.item(),
                    "complex_loss": complexity_loss,
                    "learning_rate": optimizer.param_groups[0]['lr']
                }
                logger.log_metrics(metrics_log, epoch * len(train_loader) + batch_idx)
                logger.log_gradients(model, epoch * len(train_loader) + batch_idx)
                logger.log_weights(model, epoch * len(train_loader) + batch_idx)
                
                # Log equation periodically
                if (epoch * len(train_loader) + batch_idx) % 1000 == 0:
                    equation = model.to_string(final=not soft_best)
                    logger.log_equation(equation, epoch * len(train_loader) + batch_idx)
        
        # Compute average losses for the epoch
        avg_train_loss = np.mean(epoch_train_losses)
        avg_reg_loss = np.mean(epoch_reg_losses)
        avg_complex_loss = np.mean(epoch_complex_losses)
        
        train_losses.append(avg_train_loss)
        reg_losses.append(avg_reg_loss)
        complex_losses.append(avg_complex_loss)
        
        # Validation Phase
        model.eval()
        val_eq_batch_results = []
        val_nn_batch_results = []
        val_eq_batch_orig = []
        val_eq_batch_complex = []
        
        with torch.no_grad():
            for val_data, val_target in val_loader:
                val_data, val_target = val_data.float(), val_target.float()
                eq_val_loss, nn_val_loss, val_complexity, eq_val_loss_original = test_function(val_data, val_target, model, metrics, SSA, soft_best)
                
                val_eq_batch_results.append(eq_val_loss)
                val_nn_batch_results.append(nn_val_loss)
                val_eq_batch_orig.append(eq_val_loss_original)
                val_eq_batch_complex.append(val_complexity)
        
        # Compute mean validation losses
        mean_eq_val_loss = np.mean(np.concatenate(val_eq_batch_results, axis=-1), axis=-1, keepdims=True)
        mean_nn_val_loss = np.mean(np.concatenate(val_nn_batch_results, axis=-1), axis=-1, keepdims=True)
        mean_val_eq_loss_orig = np.mean(np.concatenate(val_eq_batch_orig, axis=-1), axis=-1, keepdims=True)
        mean_val_eq_loss_complex = np.mean(np.concatenate(val_eq_batch_complex, axis=-1), axis=-1, keepdims=True)
        
        val_eq_losses.append(mean_eq_val_loss.tolist())
        val_nn_losses.append(mean_nn_val_loss.tolist())
        val_eq_loss_orig.append(mean_val_eq_loss_orig.tolist())
        val_eq_loss_complex.append(mean_val_eq_loss_complex.tolist())
        
        # Determine competition value
        compete_value = mean_eq_val_loss[0] if not SSA else mean_nn_val_loss[0]
        
        # Model Selection
        if get_best_function(compete_value, best_value):
            best_model = deepcopy(model)
            best_value = compete_value
            best_param_n = mean_eq_val_loss[-1]
            early_counter = 0
        else:
            early_counter += 1
            if early_counter > early_stop_patience and not SSA:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Logging validation metrics
        if logger:
            val_metrics_log = {
                "val_eq_loss": mean_eq_val_loss.item(),
                "val_nn_loss": mean_nn_val_loss.item(),
                "val_complex_loss": mean_val_eq_loss_complex.item()
            }
            logger.log_metrics(val_metrics_log, epoch)
        
        # Optionally adjust learning rates or other hyperparameters here
    
    # Compile results
    result_dict = {
        "model": best_model.to_string(final=not soft_best),
        "best_value": best_value.tolist(),
        "best_param_n": best_param_n.tolist(),
        "train_loss": train_losses,
        "reg_loss": reg_losses,
        "complex_loss": complex_losses,
        "val_eq_loss": val_eq_losses,
        "val_nn_loss": val_nn_losses,
        "val_eq_loss_original": val_eq_loss_orig,
        "val_eq_loss_complex": val_eq_loss_complex,
    }
    
    if not SSA:
        result_dict["final_model"] = best_model
        result_dict["soft_model"] = best_model.to_string()
        print(best_model.to_string(final=not soft_best))
    
    return result_dict

def test_function(val_data, val_target, model, metrics, SSA, soft_best):
    """
    Evaluates the model on validation data.
    
    Parameters:
        val_data (Tensor): Input data for validation.
        val_target (Tensor): Target labels for validation.
        model (nn.Module): The trained model.
        metrics (list of functions): List of metric functions to evaluate.
        SSA (bool): Flag for selecting validation metric.
        soft_best (bool): Flag for soft model selection.
    
    Returns:
        tuple: (eq_val_loss, nn_val_loss, complexity, eq_val_loss_original)
    """
    inputs, labels = val_data.float(), val_target.float()
    if not SSA:
        model.eval()
    predicted_network = model(inputs).cpu().detach().numpy()
    string_res = model.to_string(final=not soft_best)
    
    if "nan" in string_res or "inf" in string_res or string_res.strip() == "":
        predicted_eq = np.nan * np.ones_like(labels)
    else:
        try:
            f = eval(f"lambda x: {string_res}")
            detached_inputs = inputs.detach().cpu().numpy()
            predicted_eq = f(detached_inputs)
        except Exception as e:
            print(f"Error evaluating expression: {e}")
            predicted_eq = np.nan * np.ones_like(labels)
    
    predicted_eq = predicted_eq[..., None] if predicted_eq.size > 1 else np.tile(predicted_eq, detached_inputs.shape[0])[..., None]
    
    # Compute metrics
    loss_eq_original = [metric(predicted_eq, labels.numpy()) for metric in metrics]
    loss_eq_original = np.array(loss_eq_original)[np.newaxis, ...]
    
    complexity = len(string_res)
    
    if len(metrics) == 1:
        loss_eq = [metric(predicted_eq, labels.numpy())[np.newaxis, ...] + 0 * complexity for metric in metrics]
    else:
        loss_eq = [metric(predicted_eq, labels.numpy())[np.newaxis, ...] for metric in metrics]
    
    n_parameters = count_parameters(string_res)
    loss_eq = np.concatenate(loss_eq + [np.array([[n_parameters if n_parameters > 0 else 100]])], axis=0)
    
    loss_network = np.concatenate([metric(predicted_network, labels.numpy())[np.newaxis, ...] + 0 * complexity for metric in metrics])
    
    return loss_eq, loss_network, complexity, loss_eq_original

def get_best_function(new_best, old_best=None, metrics_optimize="min"):
    """
    Determines if the current model is better than the previous best.
    
    Parameters:
        new_best (float): Current validation metric.
        old_best (float, optional): Previous best validation metric.
        metrics_optimize (str): "min" or "max".
    
    Returns:
        bool: True if current model is better, else False.
    """
    if old_best is None:
        return True
    
    if metrics_optimize == "min":
        return new_best < old_best
    else:
        return new_best > old_best

def count_parameters(string_res):
    """
    Counts the number of parameters in the symbolic expression.
    
    Parameters:
        string_res (str): Symbolic expression as a string.
    
    Returns:
        int: Number of parameters.
    """
    # Implement your parameter counting logic based on the expression
    # Placeholder implementation:
    try:
        # Example: Count unique variables/constants
        tokens = string_res.replace('(', ' ').replace(')', ' ').split()
        unique_tokens = set(tokens)
        return len(unique_tokens)
    except:
        return 0
