import numpy as np
import pointpats
import pointpats.distance_statistics as dstats
import random
from scipy.stats import norm

def in_prob(mean, std_dev, time_slice, end_time):
    """
    Calculate the inclusion probability of a point being included in a time slice.

    Parameters:
    mean (float): Mean of the Gaussian distribution.
    std_dev (float): Standard deviation of the Gaussian distribution.
    time_slice (float): The time slice for which to calculate the probability.
    end_time (float): The end time for the calculation.

    Returns:
    float: The inclusion probability.
    """
    density_func = norm(loc=mean, scale=std_dev)
    integral = density_func.cdf(time_slice)
    total_integral = density_func.cdf(end_time)
    return integral / total_integral

def in_probs(points, time_slices, end_time):
    """
    Precompute the inclusion probabilities for all points across all time slices.

    Parameters:
    points (array-like): List of points with [x, y, mean, std_dev].
    time_slices (array-like): Array of time slices.
    end_time (float): The end time for the calculation.

    Returns:
    np.ndarray: A 2D array of inclusion probabilities.
    """
    n_points = len(points)
    n_slices = len(time_slices)
    inclusion_probs = np.zeros((n_points, n_slices))
    
    for i, (x, y, mean, std_dev) in enumerate(points):
        density_func = norm(loc=mean, scale=std_dev)
        total_integral = density_func.cdf(end_time)
        for j, t in enumerate(time_slices):
            integral = density_func.cdf(t)
            inclusion_probs[i, j] = integral / total_integral
    
    return inclusion_probs

def mc_samples(points, time_slices, inclusion_probs, num_iterations=1000):
    """
    Generate Monte Carlo samples for each time slice based on precomputed inclusion probabilities.

    Parameters:
    points (array-like): List of points with [x, y, mean, std_dev].
    time_slices (array-like): Array of time slices.
    inclusion_probs (np.ndarray): Precomputed inclusion probabilities.
    num_iterations (int): Number of Monte Carlo iterations.

    Returns:
    list: A list of lists where each sublist contains tuples of time slice and included points.
    """
    point_sets = []

    for _ in range(num_iterations):
        iteration_set = []
        for j, t in enumerate(time_slices):
            included_points = np.array([[p[0], p[1]] for p, prob in zip(points, inclusion_probs[:, j]) if random.random() < prob])
            iteration_set.append((t, included_points))
        point_sets.append(iteration_set)
    
    return point_sets

def temporal_k(point_sets, distances, num_iterations, time_slices):
    num_distances = len(distances)
    num_slices = len(time_slices)
    k_results = np.zeros((num_distances, num_slices, num_iterations))

    for iteration_index, iteration_set in enumerate(point_sets):
        for t_index, (t, points) in enumerate(iteration_set):
            if len(points) < 2:
                continue
            
            coordinates = np.array(points)
            support, k_estimate = dstats.k(coordinates, support=distances)
            k_normalized = k_estimate / (np.pi * distances ** 2)  # Normalizing by expected CSR value
            k_results[:, t_index, iteration_index] = k_estimate
    
    return k_results

def temporal_pcor(k_results, distances):
    """
    Estimate the pair correlation function g(d) from Ripley's K function.

    Parameters:
    k_results (np.ndarray): A 3D array where the first dimension is the distance, 
                            the second dimension is the time slice, and the third dimension is the iteration.
    distances (array-like): Array of distances at which K was calculated.

    Returns:
    np.ndarray: A 3D array of the same shape as k_results, containing the pair correlation function g(d).
    """
    g_results = np.gradient(k_results, axis=0) / (2 * np.pi * distances[:, np.newaxis, np.newaxis])
    return g_results