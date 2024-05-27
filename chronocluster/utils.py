import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def clustering_heatmap(results,
                       distances,
                       time_slices,
                       result_type = 'K'):
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

def plot_derivative(results, 
                    distances, 
                    time_slices, 
                    t_index = 0, 
                    result_type='K'):
    """
    Plot the derivative of Ripley's K or L function over distance for a specific time slice.
    
    Parameters:
    results (np.ndarray): A 3D array where the first dimension is the distance, 
                          the second dimension is the time slice, and the third dimension is the iteration.
    distances (array-like): Array of distances at which K or L was calculated.
    time_slices (array-like): Array of time slices.
    t_index (int): Index of the time slice desired for plotting.
    result_type (str): The type of results being plotted ('K' for Ripley's K function, 'L' for Ripley's L function).
    """
    mean_values = np.mean(results[:, t_index, :], axis=1)
    derivative_values = np.gradient(mean_values, distances)

    plt.figure(figsize=(8, 6))
    plt.plot(distances, derivative_values, label=f'd{result_type}/dd')
    plt.xlabel('Distance')
    plt.ylabel(f'd{result_type}/dd')
    plt.title(f"Derivative of {result_type}(d) for Time Slice {time_slices[t_index]}")
    plt.legend()
    plt.show()
    
def rotate_matrix(results, distances):
    """
    Rotate the 3D results matrix by 45 degrees.

    Parameters:
    results (np.ndarray): A 3D array where the first dimension is the distance, 
                          the second dimension is the time slice, and the third dimension is the iteration.

    Returns:
    np.ndarray: Rotated 3D results matrix.
    """
    num_distances, num_slices, num_iterations = results.shape
    rotated_results = np.zeros_like(results)
    
    for i in range(num_distances):
        for j in range(num_slices):
            for k in range(num_iterations):
                # Apply the rotation matrix
                original_value = results[i, j, k]
                rotated_x = (original_value - distances[i]) / np.sqrt(2)
                rotated_y = (original_value + distances[i]) / np.sqrt(2)
                rotated_results[i, j, k] = rotated_y  # Use rotated_y for the new value
    
    return rotated_results

def plot_l_diff(l_results, distances, time_slices):
    """
    Plot the difference d - L(d) over distance for a specific time slice.
    
    Parameters:
    l_results (np.ndarray): A 3D array of L function values.
    distances (array-like): Array of distances at which L was calculated.
    time_slices (array-like): Array of time slices.
    """
    t_index = 5  # For example, the 6th time slice
    mean_l_values = np.mean(l_results[:, t_index, :], axis=1)
    l_diff = mean_l_values - distances

    plt.figure(figsize=(8, 6))
    plt.plot(distances, l_diff, label='d - L(d)')
    plt.axhline(0, color='gray', linestyle='--', label='Reference Line')
    plt.xlabel('Distance (d)')
    plt.ylabel('d - L(d)')
    plt.title(f"d - L(d) for Time Slice {time_slices[t_index]}")
    plt.legend()
    plt.show()
    
def plot_mc_points(simulations, 
                   iter = 0, 
                   t = 0):
    """
    Plot points from the simulations for a given iteration and time slice.
    
    Parameters:
    simulations (list): List of simulated point sets from mc_samples.
    iter (int): Index of the simulation iteration to plot.
    t (int): Index of the time slice to plot.
    """
    time_slice, points = simulations[iter][t]
    
    if points.shape == (0,):
        raise NoPointsInTimeSliceException(f"No points available in iteration {iter}, time slice {t}.")

    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], alpha=0.6)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Simulation Points for Iteration {iter} and Time Slice {t}')
    plt.show()

# Exceptions

class NoPointsInTimeSliceException(Exception):
    """Exception raised when the selected time slice has no points."""
    def __init__(self, message="No points in the selected time slice."):
        self.message = message
        super().__init__(self.message)
