import numpy as np
import matplotlib.pyplot as plt

def plot_results(x_values, y_values, predictions):
    """Plot the true function and model predictions."""
    # Convert to numpy and flatten if needed
    x_np = x_values.cpu().numpy().flatten()
    y_np = y_values.cpu().numpy().flatten()
    pred_np = predictions.cpu().numpy().flatten()
    
    # Sort points for better visualization
    sort_idx = np.argsort(x_np)
    x_np = x_np[sort_idx]
    y_np = y_np[sort_idx]
    pred_np = pred_np[sort_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_np, y_np, label='True Function')
    plt.plot(x_np, pred_np, '--', label='EQL Prediction')
    plt.legend()
    plt.title('EQL Function Learning Results - Nguyen-1')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.pause(0.001)
    plt.show() 