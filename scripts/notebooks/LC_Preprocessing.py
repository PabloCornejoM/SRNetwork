# Light Curve Preprocessing for Symbolic Regression
# This script contains utilities for preprocessing astronomical light curve data
# before applying symbolic regression models.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from sklearn.preprocessing import StandardScaler
import random
from tqdm import tqdm

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

def load_light_curve(file_path):
    """
    Load a light curve from a file (supports CSV format)
    
    Expected columns: 'time', 'flux', 'flux_err', 'filter' (optional)
    
    Parameters:
    -----------
    file_path : str
        Path to the light curve file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the light curve data
    """
    # Load the data based on file type
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.pkl'):
        df = pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Basic validation
    required_cols = ['time', 'flux']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in light curve data")
    
    # If flux_err is not present, add a placeholder
    if 'flux_err' not in df.columns:
        df['flux_err'] = np.abs(df['flux']) * 0.05  # 5% error as placeholder
        
    return df

def visualize_light_curve(lc, title=None, ax=None):
    """
    Visualize a light curve with error bars
    
    Parameters:
    -----------
    lc : pd.DataFrame
        Light curve data with 'time', 'flux', 'flux_err' columns
    title : str or None
        Title for the plot
    ax : matplotlib.axes or None
        Axes to plot on, if None, a new figure is created
        
    Returns:
    --------
    matplotlib.axes
        The axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by time to ensure connected lines make sense
    if not lc.empty:
        lc_sorted = lc.sort_values('time')
        
        # Plot the light curve
        ax.errorbar(lc_sorted['time'], lc_sorted['flux'], 
                   yerr=lc_sorted['flux_err'] if 'flux_err' in lc_sorted.columns else None,
                   fmt='o-', markersize=4, capsize=2, alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Flux')
    if title:
        ax.set_title(title)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return ax

def compare_light_curves(original_lc, modified_lc, titles=None, figsize=(12, 6)):
    """
    Compare original and modified light curves side by side
    
    Parameters:
    -----------
    original_lc : pd.DataFrame
        Original light curve
    modified_lc : pd.DataFrame
        Modified light curve
    titles : tuple or None
        Titles for the two plots (original, modified)
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure
        The figure with the plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Set default titles if not provided
    if titles is None:
        titles = ('Original Light Curve', 'Modified Light Curve')
    
    # Plot original
    visualize_light_curve(original_lc, titles[0], ax1)
    
    # Plot modified
    visualize_light_curve(modified_lc, titles[1], ax2)
    
    # Ensure the y-axis limits are the same for both plots for easier comparison
    y_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    y_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    return fig

def add_noise(lc, noise_level=0.1, noise_type='gaussian'):
    """
    Add noise to a light curve
    
    Parameters:
    -----------
    lc : pd.DataFrame
        Light curve data with 'time', 'flux', 'flux_err' columns
    noise_level : float
        Level of noise to add (as a fraction of flux)
    noise_type : str
        Type of noise to add ('gaussian', 'uniform', or 'proportional')
        
    Returns:
    --------
    pd.DataFrame
        Light curve with added noise
    """
    lc_noisy = lc.copy()
    
    if noise_type == 'gaussian':
        # Add Gaussian noise with standard deviation proportional to the flux value
        noise = np.random.normal(0, noise_level * np.abs(lc['flux']), size=len(lc))
    elif noise_type == 'uniform':
        # Add uniform noise
        noise = np.random.uniform(-noise_level * np.abs(lc['flux']), 
                                 noise_level * np.abs(lc['flux']), 
                                 size=len(lc))
    elif noise_type == 'proportional':
        # Add noise proportional to flux value
        noise = np.random.normal(0, 1, size=len(lc)) * noise_level * np.abs(lc['flux'])
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    
    lc_noisy['flux'] = lc['flux'] + noise
    
    # Update error bars to reflect the added noise
    if 'flux_err' in lc.columns:
        lc_noisy['flux_err'] = np.sqrt(lc['flux_err']**2 + (noise_level * np.abs(lc['flux']))**2)
    
    return lc_noisy

def create_sparse_lc(lc, sparsity_level=0.5, method='random'):
    """
    Create a sparse version of a light curve by removing data points
    
    Parameters:
    -----------
    lc : pd.DataFrame
        Light curve data
    sparsity_level : float
        Fraction of points to remove (0.0-1.0)
    method : str
        Method for creating sparsity ('random', 'uniform', 'peak_preserving')
        
    Returns:
    --------
    pd.DataFrame
        Sparse light curve
    """
    lc_sparse = lc.copy()
    n_points = len(lc)
    n_remove = int(n_points * sparsity_level)
    
    if n_remove >= n_points:
        raise ValueError(f"Sparsity level {sparsity_level} would remove all points")
    
    if method == 'random':
        # Randomly remove points
        indices_to_keep = np.random.choice(n_points, size=n_points-n_remove, replace=False)
        lc_sparse = lc.iloc[sorted(indices_to_keep)].reset_index(drop=True)
    
    elif method == 'uniform':
        # Keep uniformly spaced points
        indices_to_keep = np.linspace(0, n_points-1, n_points-n_remove, dtype=int)
        lc_sparse = lc.iloc[indices_to_keep].reset_index(drop=True)
    
    elif method == 'peak_preserving':
        # Identify the peak (max absolute flux)
        peak_idx = np.argmax(np.abs(lc['flux']))
        
        # Always keep the peak and points around it
        peak_window = int(n_points * 0.1)  # 10% of points around peak
        start_idx = max(0, peak_idx - peak_window//2)
        end_idx = min(n_points, peak_idx + peak_window//2)
        peak_indices = list(range(start_idx, end_idx))
        
        # Randomly select from remaining points
        other_indices = list(set(range(n_points)) - set(peak_indices))
        n_other_to_keep = (n_points - n_remove) - len(peak_indices)
        
        if n_other_to_keep > 0:
            selected_other = np.random.choice(other_indices, size=n_other_to_keep, replace=False)
            indices_to_keep = sorted(list(peak_indices) + list(selected_other))
        else:
            # If we need to remove points from peak region too
            indices_to_keep = np.random.choice(peak_indices, size=n_points-n_remove, replace=False)
            
        lc_sparse = lc.iloc[indices_to_keep].reset_index(drop=True)
    
    else:
        raise ValueError(f"Unsupported sparsity method: {method}")
    
    return lc_sparse

def combine_light_curves(lc_list, method='sum', interpolation=None):
    """
    Combine multiple light curves into one
    
    Parameters:
    -----------
    lc_list : list of pd.DataFrame
        List of light curves to combine
    method : str
        Method for combining ('sum', 'average', 'weighted_average')
    interpolation : str or None
        Interpolation method to use for aligning time points ('linear', 'cubic', None)
        
    Returns:
    --------
    pd.DataFrame
        Combined light curve
    """
    if not lc_list:
        raise ValueError("Empty list of light curves provided")
    
    if len(lc_list) == 1:
        return lc_list[0].copy()
    
    # First, determine the time grid for the combined light curve
    if interpolation is None:
        # Without interpolation, use the union of all time points
        all_times = np.concatenate([lc['time'].values for lc in lc_list])
        unique_times = np.sort(np.unique(all_times))
    else:
        # With interpolation, find the min and max times across all curves
        min_time = min(lc['time'].min() for lc in lc_list)
        max_time = max(lc['time'].max() for lc in lc_list)
        
        # Create a regular grid within this range
        # Use the median cadence from all curves to determine spacing
        cadences = []
        for lc in lc_list:
            times = np.sort(lc['time'].values)
            cadences.extend(np.diff(times))
        
        if cadences:
            median_cadence = np.median(cadences)
            unique_times = np.arange(min_time, max_time + median_cadence, median_cadence)
        else:
            # If we couldn't determine cadence, use 100 points
            unique_times = np.linspace(min_time, max_time, 100)
    
    # Initialize the combined flux and error arrays
    combined_flux = np.zeros(len(unique_times))
    combined_flux_err = np.zeros(len(unique_times))
    
    # For weighted average
    if method == 'weighted_average':
        weights_sum = np.zeros(len(unique_times))
    
    # Process each light curve
    for lc in lc_list:
        # If interpolation is requested, interpolate this light curve onto the combined time grid
        if interpolation is not None:
            if interpolation == 'linear':
                interp_func = interpolate.interp1d(lc['time'], lc['flux'], 
                                                  bounds_error=False, fill_value=0)
            elif interpolation == 'cubic':
                if len(lc) > 3:  # Cubic requires at least 4 points
                    interp_func = interpolate.interp1d(lc['time'], lc['flux'], kind='cubic',
                                                      bounds_error=False, fill_value=0)
                else:
                    interp_func = interpolate.interp1d(lc['time'], lc['flux'], 
                                                      bounds_error=False, fill_value=0)
            else:
                raise ValueError(f"Unsupported interpolation method: {interpolation}")
                
            # Interpolate flux at unique times
            interpolated_flux = interp_func(unique_times)
            
            # For errors, we'll use a simpler approach (linear interpolation or nearest)
            if 'flux_err' in lc.columns:
                err_interp = interpolate.interp1d(lc['time'], lc['flux_err'], 
                                                 bounds_error=False, fill_value=np.nan)
                interpolated_err = err_interp(unique_times)
            else:
                interpolated_err = np.abs(interpolated_flux) * 0.05  # 5% placeholder
                
            # Replace NaNs with zeros for flux, and high values for errors
            interpolated_flux = np.nan_to_num(interpolated_flux, nan=0.0)
            interpolated_err = np.nan_to_num(interpolated_err, nan=1e10)
            
            if method == 'sum':
                combined_flux += interpolated_flux
                combined_flux_err = np.sqrt(combined_flux_err**2 + interpolated_err**2)
            elif method == 'average':
                combined_flux += interpolated_flux
                combined_flux_err += interpolated_err**2
            elif method == 'weighted_average':
                # Weights are inversely proportional to squared errors
                weights = 1.0 / (interpolated_err**2)
                weights[~np.isfinite(weights)] = 0.0  # Handle infinite or NaN weights
                
                combined_flux += interpolated_flux * weights
                weights_sum += weights
        else:
            # Without interpolation, add flux values only at matching time points
            for i, t in enumerate(unique_times):
                matches = lc['time'] == t
                if np.any(matches):
                    if method == 'sum':
                        combined_flux[i] += lc.loc[matches, 'flux'].values[0]
                        if 'flux_err' in lc.columns:
                            err = lc.loc[matches, 'flux_err'].values[0]
                            combined_flux_err[i] = np.sqrt(combined_flux_err[i]**2 + err**2)
                    elif method == 'average':
                        combined_flux[i] += lc.loc[matches, 'flux'].values[0]
                        if 'flux_err' in lc.columns:
                            combined_flux_err[i] += lc.loc[matches, 'flux_err'].values[0]**2
                    elif method == 'weighted_average':
                        if 'flux_err' in lc.columns:
                            err = lc.loc[matches, 'flux_err'].values[0]
                            weight = 1.0 / (err**2) if err > 0 else 0.0
                            combined_flux[i] += lc.loc[matches, 'flux'].values[0] * weight
                            weights_sum[i] += weight
    
    # Finalize the combined light curve based on the method
    if method == 'average':
        # Count how many light curves contributed to each time point
        counts = np.zeros(len(unique_times))
        for lc in lc_list:
            for i, t in enumerate(unique_times):
                if np.any(lc['time'] == t):
                    counts[i] += 1
        
        # Avoid division by zero
        valid_counts = counts > 0
        combined_flux[valid_counts] /= counts[valid_counts]
        combined_flux_err[valid_counts] = np.sqrt(combined_flux_err[valid_counts]) / counts[valid_counts]
    
    elif method == 'weighted_average':
        # Normalize by sum of weights
        valid_weights = weights_sum > 0
        combined_flux[valid_weights] /= weights_sum[valid_weights]
        combined_flux_err[valid_weights] = 1.0 / np.sqrt(weights_sum[valid_weights])
    
    # Create the combined DataFrame
    combined_lc = pd.DataFrame({
        'time': unique_times,
        'flux': combined_flux,
        'flux_err': combined_flux_err
    })
    
    return combined_lc

def create_sample_light_curve(n_points=100, curve_type='supernova'):
    """
    Create a synthetic light curve for demonstration
    
    Parameters:
    -----------
    n_points : int
        Number of data points
    curve_type : str
        Type of curve to generate ('supernova', 'variable_star', 'transit')
        
    Returns:
    --------
    pd.DataFrame
        Synthetic light curve data
    """
    # Time values (evenly spaced for simplicity)
    time = np.linspace(0, 100, n_points)
    
    if curve_type == 'supernova':
        # Simple model of supernova light curve (rise and decline)
        peak_time = 30
        rise_rate = 0.3
        decline_rate = 0.1
        
        flux = np.zeros(n_points)
        for i, t in enumerate(time):
            if t < peak_time:
                flux[i] = np.exp(rise_rate * (t - peak_time))
            else:
                flux[i] = np.exp(-decline_rate * (t - peak_time))
                
    elif curve_type == 'variable_star':
        # Sinusoidal variation with some complexity
        period1 = 10
        period2 = 23
        flux = 1 + 0.5 * np.sin(2 * np.pi * time / period1) + 0.3 * np.sin(2 * np.pi * time / period2)
        
    elif curve_type == 'transit':
        # Star with periodic transits
        flux = np.ones(n_points)
        transit_period = 20
        transit_duration = 2
        transit_depth = 0.1
        
        for i, t in enumerate(time):
            transit_phase = (t % transit_period) / transit_period
            if transit_phase < transit_duration / transit_period:
                flux[i] -= transit_depth * (1 - (transit_phase * transit_period / (transit_duration/2) - 1)**2)
                flux[i] = max(flux[i], 1 - transit_depth)
    else:
        raise ValueError(f"Unsupported curve type: {curve_type}")
    
    # Add a small amount of intrinsic noise
    base_noise = 0.02 * np.max(flux) * np.random.normal(0, 1, n_points)
    flux += base_noise
    
    # Create error values (proportional to flux)
    flux_err = 0.05 * np.abs(flux)
    
    # Create DataFrame
    return pd.DataFrame({'time': time, 'flux': flux, 'flux_err': flux_err})

def evaluate_noise_impact(original_lc, noise_levels, noise_types, sr_model_func, n_repeats=3):
    """
    Evaluate the impact of different noise levels on SR model performance
    
    Parameters:
    -----------
    original_lc : pd.DataFrame
        Original light curve data
    noise_levels : list
        List of noise levels to test
    noise_types : list
        List of noise types to test
    sr_model_func : function
        Function that takes (x, y) data and returns (equation, r2_score)
    n_repeats : int
        Number of times to repeat each experiment for statistical significance
        
    Returns:
    --------
    pd.DataFrame
        Results with noise levels, types, equations, and performance metrics
    """
    results = []
    
    # Prepare original data for the SR model
    x_original = original_lc['time'].values
    y_original = original_lc['flux'].values
    
    # Baseline performance on original data
    try:
        baseline_eq, baseline_r2 = sr_model_func(x_original, y_original)
        results.append({
            'noise_type': 'none',
            'noise_level': 0.0,
            'equation': baseline_eq,
            'r2_score': baseline_r2,
            'repeat': 0
        })
    except Exception as e:
        print(f"Error in baseline evaluation: {e}")
    
    # Test different noise levels and types
    for noise_type in noise_types:
        for noise_level in noise_levels:
            for repeat in range(n_repeats):
                try:
                    # Apply noise
                    noisy_lc = add_noise(original_lc, noise_level=noise_level, noise_type=noise_type)
                    
                    # Prepare data for the SR model
                    x_noisy = noisy_lc['time'].values
                    y_noisy = noisy_lc['flux'].values
                    
                    # Run SR model
                    equation, r2_score = sr_model_func(x_noisy, y_noisy)
                    
                    # Store results
                    results.append({
                        'noise_type': noise_type,
                        'noise_level': noise_level,
                        'equation': equation,
                        'r2_score': r2_score,
                        'repeat': repeat
                    })
                except Exception as e:
                    print(f"Error with {noise_type} noise at level {noise_level}, repeat {repeat}: {e}")
    
    return pd.DataFrame(results)

def evaluate_sparsity_impact(original_lc, sparsity_levels, sparsity_methods, sr_model_func, n_repeats=3):
    """
    Evaluate the impact of different sparsity levels on SR model performance
    
    Parameters:
    -----------
    original_lc : pd.DataFrame
        Original light curve data
    sparsity_levels : list
        List of sparsity levels to test
    sparsity_methods : list
        List of sparsity methods to test
    sr_model_func : function
        Function that takes (x, y) data and returns (equation, r2_score)
    n_repeats : int
        Number of times to repeat each experiment for statistical significance
        
    Returns:
    --------
    pd.DataFrame
        Results with sparsity levels, methods, equations, and performance metrics
    """
    results = []
    
    # Prepare original data for the SR model
    x_original = original_lc['time'].values
    y_original = original_lc['flux'].values
    
    # Baseline performance on original data
    try:
        baseline_eq, baseline_r2 = sr_model_func(x_original, y_original)
        results.append({
            'sparsity_method': 'none',
            'sparsity_level': 0.0,
            'n_points': len(original_lc),
            'equation': baseline_eq,
            'r2_score': baseline_r2,
            'repeat': 0
        })
    except Exception as e:
        print(f"Error in baseline evaluation: {e}")
    
    # Test different sparsity levels and methods
    for method in sparsity_methods:
        for level in sparsity_levels:
            for repeat in range(n_repeats):
                try:
                    # Create sparse light curve
                    sparse_lc = create_sparse_lc(original_lc, sparsity_level=level, method=method)
                    
                    # Prepare data for the SR model
                    x_sparse = sparse_lc['time'].values
                    y_sparse = sparse_lc['flux'].values
                    
                    # Run SR model
                    equation, r2_score = sr_model_func(x_sparse, y_sparse)
                    
                    # Store results
                    results.append({
                        'sparsity_method': method,
                        'sparsity_level': level,
                        'n_points': len(sparse_lc),
                        'equation': equation,
                        'r2_score': r2_score,
                        'repeat': repeat
                    })
                except Exception as e:
                    print(f"Error with {method} sparsity at level {level}, repeat {repeat}: {e}")
    
    return pd.DataFrame(results)

def evaluate_combined_effects(original_lc, sparsity_levels, noise_levels, sr_model_func):
    """
    Evaluate the combined effects of sparsity and noise on SR model performance
    
    Parameters:
    -----------
    original_lc : pd.DataFrame
        Original light curve
    sparsity_levels : list
        Sparsity levels to test
    noise_levels : list
        Noise levels to test
    sr_model_func : function
        SR model function
        
    Returns:
    --------
    pd.DataFrame
        Results with both factors
    """
    results = []
    
    for sparsity_level in sparsity_levels:
        for noise_level in noise_levels:
            try:
                # Apply transformations
                processed_lc = original_lc.copy()
                
                # Apply sparsity if needed
                if sparsity_level > 0.0:
                    processed_lc = create_sparse_lc(processed_lc, sparsity_level=sparsity_level, method='random')
                
                # Apply noise if needed
                if noise_level > 0.0:
                    processed_lc = add_noise(processed_lc, noise_level=noise_level, noise_type='gaussian')
                
                # Run SR model
                equation, r2_score = sr_model_func(processed_lc['time'].values, processed_lc['flux'].values)
                
                # Store results
                results.append({
                    'sparsity_level': sparsity_level,
                    'noise_level': noise_level,
                    'n_points': len(processed_lc),
                    'equation': equation,
                    'r2_score': r2_score
                })
            except Exception as e:
                print(f"Error with sparsity {sparsity_level}, noise {noise_level}: {e}")
    
    return pd.DataFrame(results)

def dummy_sr_model(x, y):
    """
    A dummy SR model for demonstration purposes. 
    Replace this with your actual SR model function.
    
    Parameters:
    -----------
    x : np.ndarray
        Input features
    y : np.ndarray
        Target values
        
    Returns:
    --------
    tuple
        (equation string, R² score)
    """
    # This is just a placeholder - replace with your actual SR model
    from sklearn.metrics import r2_score
    import time
    
    # Simulate fitting process
    time.sleep(0.1)  
    
    # Dummy equation and score that degrades with noise
    noise_level = np.std(y) / np.mean(np.abs(y)) - 0.05  # estimate noise from data
    noise_level = max(0, min(1, noise_level))  # clamp to [0,1]
    
    if noise_level < 0.1:
        eq = "a * exp(-b * (x - c)^2) + d"
        r2 = 0.95 - noise_level * 2
    elif noise_level < 0.3:
        eq = "a * exp(-b * x) + c"
        r2 = 0.85 - noise_level
    else:
        eq = "a * x + b"
        r2 = 0.7 - noise_level
    
    return eq, r2

if __name__ == "__main__":
    # Example usage
    # Create a sample light curve
    lc = create_sample_light_curve(n_points=100, curve_type='supernova')
    
    # Visualize original light curve
    plt.figure(figsize=(10, 6))
    visualize_light_curve(lc, "Original Light Curve")
    plt.show()
    
    # Add noise and visualize
    noisy_lc = add_noise(lc, noise_level=0.1, noise_type='gaussian')
    compare_light_curves(lc, noisy_lc, ("Original", "With Gaussian Noise"))
    plt.show()
    
    # Create sparse version and visualize
    sparse_lc = create_sparse_lc(lc, sparsity_level=0.5, method='random')
    compare_light_curves(lc, sparse_lc, ("Original", "With 50% Random Sparsity"))
    plt.show()
    
    # Combine multiple light curves
    lc2 = create_sample_light_curve(n_points=100, curve_type='supernova')
    combined_lc = combine_light_curves([lc, lc2], method='average', interpolation='linear')
    compare_light_curves(lc, combined_lc, ("Original", "Combined with Another Light Curve"))
    plt.show() 