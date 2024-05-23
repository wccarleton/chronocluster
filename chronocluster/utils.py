import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def clustering_heatmap(results, time_slices, distances, result_type='K'):
    """
    Plot a heatmap of either Ripley's K function or the pair correlation function over time and distance.
    
    Parameters:
    results (np.ndarray): A 3D array where the first dimension is the distance, 
                          the second dimension is the time slice, and the third dimension is the iteration.
    time_slices (array-like): Array of time slices.
    distances (array-like): Array of distances at which K or g was calculated.
    result_type (str): The type of results being plotted ('K' for Ripley's K function or 'g' for pair correlation function).
    """
    mean_values = np.mean(results, axis=2)
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(mean_values, xticklabels=time_slices, yticklabels=distances, cmap='viridis', cbar_kws={'label': f"Mean {result_type}(d)"})
    plt.xlabel('Time Slices')
    plt.ylabel('Distances')
    plt.title(f"Heatmap of Mean {result_type}(d) Function Over Time and Distance")
    plt.show()