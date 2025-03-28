import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_results(x_values, y_values, predictions):
    """
    Plot the true function and model predictions.
    
    Args:
        x_values: Input values (can be 1D or 2D)
        y_values: True output values
        predictions: Model predictions
    """
    # Convert to numpy arrays
    x_np = x_values.cpu().numpy()
    y_np = y_values.cpu().numpy().flatten()  # Ensure y is flattened
    pred_np = predictions.cpu().numpy().flatten()  # Ensure predictions are flattened
    
    # Check input dimensionality
    if x_np.ndim > 1 and x_np.shape[1] > 1:
        # 2D or higher dimensional input
        # Create a figure with 2x2 subplots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. True function (3D)
        ax1 = fig.add_subplot(221, projection='3d')
        scatter1 = ax1.scatter(x_np[:, 0], x_np[:, 1], y_np, c=y_np, 
                             cmap='viridis', s=20)
        fig.colorbar(scatter1, ax=ax1, label='True Values')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_zlabel('y')
        ax1.set_title('True Function (3D)')
        
        # 2. Predicted function (3D)
        ax2 = fig.add_subplot(222, projection='3d')
        scatter2 = ax2.scatter(x_np[:, 0], x_np[:, 1], pred_np, c=pred_np, 
                             cmap='viridis', s=20)
        fig.colorbar(scatter2, ax=ax2, label='Predicted Values')
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('y')
        ax2.set_title('SRNet Prediction (3D)')
        
        # 3. Error plot (2D)
        ax3 = fig.add_subplot(223)
        error = np.abs(y_np - pred_np)
        scatter3 = ax3.scatter(x_np[:, 0], x_np[:, 1], c=error, 
                             cmap='RdYlBu_r', s=20)
        fig.colorbar(scatter3, ax=ax3, label='Absolute Error')
        ax3.set_xlabel('x1')
        ax3.set_ylabel('x2')
        ax3.set_title('Prediction Error')
        
        # 4. True vs Predicted scatter
        ax4 = fig.add_subplot(224)
        ax4.scatter(y_np, pred_np, alpha=0.5, s=20)
        max_val = max(y_np.max(), pred_np.max())
        min_val = min(y_np.min(), pred_np.min())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax4.set_xlabel('True Values')
        ax4.set_ylabel('Predicted Values')
        ax4.set_title('True vs Predicted')
        ax4.legend()
        
        # Optional: Add smooth surface plot if points are on a grid
        try:
            # Check if points form a grid
            unique_x1 = np.unique(x_np[:, 0])
            unique_x2 = np.unique(x_np[:, 1])
            if len(unique_x1) * len(unique_x2) == len(x_np):
                # Create meshgrid
                X1, X2 = np.meshgrid(unique_x1, unique_x2)
                Y_true = y_np.reshape(len(unique_x2), len(unique_x1))
                Y_pred = pred_np.reshape(len(unique_x2), len(unique_x1))
                
                # Add surface plots
                ax1.plot_surface(X1, X2, Y_true, alpha=0.3, cmap='viridis')
                ax2.plot_surface(X1, X2, Y_pred, alpha=0.3, cmap='viridis')
        except Exception as e:
            print(f"Note: Could not create surface plot. Points might not form a regular grid.")
        
    else:
        # 1D input
        # Flatten arrays
        x_np = x_np.flatten()
        
        # Sort points for better visualization
        sort_idx = np.argsort(x_np)
        x_np = x_np[sort_idx]
        y_np = y_np[sort_idx]
        pred_np = pred_np[sort_idx]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Function plot
        ax1.plot(x_np, y_np, label='True Function')
        ax1.plot(x_np, pred_np, '--', label='SRNet Prediction')
        ax1.legend()
        ax1.set_title('SRNet Function Learning Results')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.grid(True)
        
        # Error plot
        error = np.abs(y_np - pred_np)
        ax2.plot(x_np, error, 'r-', label='Absolute Error')
        ax2.set_title('Prediction Error')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Absolute Error')
        ax2.grid(True)
        ax2.legend()
    
    plt.tight_layout()
    plt.show() 