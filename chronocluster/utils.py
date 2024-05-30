import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rv_continuous

class ddelta(rv_continuous):
    """Probability functions approximating the Dirac Delta"""
    
    def __init__(self, d):
        super().__init__(name='ddelta')
        self.d = d
        self.dist = self
        self.badvalue = np.nan
        self.a = d
        self.b = d
        self.xtol = 1e-14
        self.moment_type = 1
        self.shapes = None
        self.numargs = 0
        self.vecentropy = np.vectorize(self._entropy)
        self.generic_moment = np.vectorize(self._moment)

    def _pdf(self, x):
        """Probability density function"""
        return np.inf if x == self.d else 0

    def _cdf(self, x):
        """Cumulative distribution function"""
        return 1.0 if x >= self.d else 0.0
    
    def _sf(self, x):
        """Survival function"""
        return 1.0 - self._cdf(x)

    def _ppf(self, q):
        """Percent point function (inverse of cdf)"""
        return self.d

    def _rvs(self, size=None, random_state=None):
        """Random variates"""
        return np.full(size, self.d)

    def mean(self):
        """Mean of the distribution"""
        return self.d
    
    def var(self):
        """Variance of the distribution"""
        return 0.0

    def std(self):
        """Standard deviation of the distribution"""
        return 0.0

    def _entropy(self, *args, **kwargs):
        """Entropy of the distribution"""
        return 0.0

    def _moment(self, n, *args, **kwargs):
        """nth moment of the distribution"""
        if n == 1:
            return self.mean()
        elif n == 2:
            return self.var()**2
        else:
            return np.nan

def _largest_divisor(n, max_divisors=10):
    'Utility function for legible axis tick mark density.'
    for i in range(max_divisors, 0, -1):
        if n % i == 0:
            return i
    return 1

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

    ax = sns.heatmap(mean_values, 
                    xticklabels = time_slices, 
                    yticklabels = distances, 
                    cmap = 'viridis', 
                    cbar_kws = {'label': f"Mean {result_type}(d)"})
    
    # Adjust x and y axis ticks using the largest divisor
    x_divisor = _largest_divisor(len(time_slices))
    y_divisor = _largest_divisor(len(distances))

    x_ticks_indices = np.arange(0, len(time_slices), len(time_slices) // x_divisor)
    y_ticks_indices = np.arange(0, len(distances), len(distances) // y_divisor)
    
    ax.set_xticks(x_ticks_indices)
    ax.set_xticklabels(np.round(time_slices[x_ticks_indices], 2))
    
    ax.set_yticks(y_ticks_indices)
    ax.set_yticklabels(np.round(distances[y_ticks_indices], 2))

    plt.xlabel('Time Slices')
    plt.ylabel('Distances')
    plt.title(f"Heatmap of Mean {result_type}(d) Function Over Time and Distance")

    plt.gca().invert_yaxis()
    plt.show()

def pdiff_heatmap(p_diff_array, time_slices, support):
    """
    Plot the heatmap of probability values.
    
    Parameters:
    p_diff_array (np.ndarray): Array of probability values with shape (distances, time_slices).
    time_slices (np.ndarray): Array of time slice values.
    support (np.ndarray): Array of pairwise distance values.
    """
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(p_diff_array, xticklabels=time_slices, yticklabels=support, cmap='viridis', cbar_kws={'label': 'P(diff > 0)'})
    plt.xlabel('Time Slices')
    plt.ylabel('Pairwise Distances')
    plt.title('Heatmap of P(diff > 0) Over Time and Distance')
    
    # Adjust x and y axis ticks using the largest divisor
    x_divisor = _largest_divisor(len(time_slices))
    y_divisor = _largest_divisor(len(support))

    x_ticks_indices = np.arange(0, len(time_slices), len(time_slices) // x_divisor)
    y_ticks_indices = np.arange(0, len(support), len(support) // y_divisor)
    
    ax.set_xticks(x_ticks_indices)
    ax.set_xticklabels(np.round(time_slices[x_ticks_indices], 2))
    
    ax.set_yticks(y_ticks_indices)
    ax.set_yticklabels(np.round(support[y_ticks_indices], 2))
    
    plt.gca().invert_yaxis()
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
