import numpy as np
import pointpats as pp
from pointpats import PointPattern
import pointpats.distance_statistics as dstats
import random
from scipy.stats import norm
from scipy.stats import norm, gaussian_kde
from scipy.spatial import distance


def in_prob(mean, 
            std_dev, 
            time_slice, 
            end_time):
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

def in_probs(points, 
             time_slices, 
             end_time):
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

def mc_samples(points, 
               time_slices, 
               inclusion_probs, 
               num_iterations = 1000):
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

def temporal_cluster(point_sets, 
                     distances, 
                     time_slices, 
                     num_iterations = 1000, 
                     calc_K = True, 
                     calc_L = False, 
                     calc_G = False):
    """
    Calculate Ripley's K, Ripley's L, and pair correlation function over time.

    Parameters:
    point_sets (list): List of point sets for each iteration.
    distances (np.ndarray): Array of distances at which to calculate the functions.
    time_slices (np.ndarray): Array of time slices.
    num_iterations (int): Number of Monte Carlo iterations. Default is 1000.
    calc_K (bool): Whether to calculate Ripley's K function. Default is True.
    calc_L (bool): Whether to calculate Ripley's L function. Default is False.
    calc_G (bool): Whether to calculate the pair correlation function. Default is False.

    Returns:
    tuple: A tuple containing the results for K, L, and/or g functions as 3D arrays.
    """
    num_distances = len(distances)
    num_slices = len(time_slices)
    
    k_results = np.zeros((num_distances, num_slices, num_iterations)) if calc_K else None
    l_results = np.zeros((num_distances, num_slices, num_iterations)) if calc_L else None
    g_results = np.zeros((num_distances, num_slices, num_iterations)) if calc_G else None

    for iteration_index, iteration_set in enumerate(point_sets):
        for t_index, (t, points) in enumerate(iteration_set):
            if len(points) < 2:
                continue
            
            coordinates = np.array(points)
            if calc_K:
                support, k_estimate = dstats.k(coordinates, support=distances)
                k_results[:, t_index, iteration_index] = k_estimate
            if calc_L:
                support, l_estimate = dstats.l(coordinates, support=distances)
                l_results[:, t_index, iteration_index] = l_estimate
            if calc_G:
                support, k_estimate = dstats.k(coordinates, support=distances)
                g_estimate = np.gradient(k_estimate, distances) / (2 * np.pi * distances)
                g_results[:, t_index, iteration_index] = g_estimate
    
    return k_results, l_results, g_results

def temporal_pairwise(simulations, 
                      time_slices, 
                      bw=1, 
                      density = False, 
                      max_distance = None):
    """
    Calculate pairwise distance densities over time.

    Parameters:
    simulations (list): List of simulated point sets from mc_samples.
    time_slices (array-like): Array of time slices.
    bw (float): Bin width for histograms or bandwidth for KDE.
    density (bool): If True, use KDE; otherwise, use histograms.

    Returns:
    np.ndarray: A 3D array where the dimensions are distances (support), time slices, and iterations.
    """
    num_slices = len(time_slices)
    num_iterations = len(simulations)
    all_distances = []

    # Loop over simulations and time slices to collect pairwise distances
    for simulation in simulations:
        distances_for_time_slices = []
        for time, points in simulation:
            if len(points) < 2:
                distances_for_time_slices.append(None)  # Use None to indicate no distances
                continue
            pairwise_distances = distance.pdist(points)
            distances_for_time_slices.append(pairwise_distances)
        all_distances.append(distances_for_time_slices)

    if max_distance is None:
        # Flatten the list of distances to determine the maximum pairwise distance for binning/KDE
        flat_distances = [dist for sublist in all_distances for dist in sublist if dist is not None]
        if not flat_distances:
            raise ValueError("No valid distances found.")
        max_distance = np.max([np.max(d) for d in flat_distances])

    # Create bins or support for KDE
    if density:
        support = np.linspace(0, max_distance, 100)
    else:
        bins = np.arange(0, max_distance + bw, bw)
        support = bins[:-1] + (bw / 2)  # Midpoints of bins

    # Initialize the results array
    pairwise_density = np.zeros((len(support), num_slices, num_iterations))

    # Calculate densities for each time slice and iteration
    for iteration_index in range(num_iterations):
        for t_index in range(num_slices):
            distances_for_slice = all_distances[iteration_index][t_index]
            if distances_for_slice is None:
                pairwise_density[:, t_index, iteration_index] = None  # Set to None if no distances
                continue
            if density:
                kde = gaussian_kde(distances_for_slice, bw_method=bw)
                pairwise_density[:, t_index, iteration_index] = kde(support)
            else:
                hist, _ = np.histogram(distances_for_slice, bins=bins, density=True)
                pairwise_density[:, t_index, iteration_index] = hist

    return pairwise_density, support

def csr_sample(points, x_min, x_max, y_min, y_max):
    """
    Generate a CSR sample by randomizing the locations of the points while keeping their temporal information intact.
    
    Parameters:
    points (list of lists): List of original points, where each element is a list [x, y, mean, std_dev].
    x_min (float): Minimum x coordinate for the randomization.
    x_max (float): Maximum x coordinate for the randomization.
    y_min (float): Minimum y coordinate for the randomization.
    y_max (float): Maximum y coordinate for the randomization.
    
    Returns:
    list of lists: List of CSR sampled points with the same structure as the input points.
    """
    n_points = len(points)
    randomized_points = []

    for _ in range(n_points):
        new_x = np.random.uniform(x_min, x_max)
        new_y = np.random.uniform(y_min, y_max)
        mean = points[_][2]
        std_dev = points[_][3]
        randomized_points.append([new_x, new_y, mean, std_dev])

    return randomized_points