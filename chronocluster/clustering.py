import numpy as np
import random
import pointpats.distance_statistics as dstats
from scipy.stats import norm, gaussian_kde, uniform
from scipy.integrate import simps
from scipy.spatial import distance

class Point:
    """
    A class to represent a point with spatial coordinates and temporal distributions.

    Attributes:
    -----------
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.
    start_distribution : scipy.stats.rv_continuous
        The probability distribution for the start date.
    end_distribution : scipy.stats.rv_continuous
        The probability distribution for the end date.

    Methods:
    --------
    __init__(self, x, y, start_distribution, end_distribution, verbose=False):
        Initializes a new Point instance with the given coordinates and distributions.

    _check_distributions(self, verbose):
        Checks the temporal consistency of the start and end distributions.

    _calculate_overlap_ratio(self):
        Calculates the overlap ratio between the start and end distributions.

    calculate_inclusion_probability(self, time_slice):
        Calculates the inclusion probability of the point for a given time slice.

    _repr_distribution(self, dist):
        Returns a string representation of the distribution.

    __repr__(self):
        Returns a string representation of the Point instance.
    """
    __slots__ = ['x', 'y', 'start_distribution', 'end_distribution']

    def __init__(self, x, y, start_distribution, end_distribution, verbose=False):
        """
        Initializes a new Point instance with the given coordinates and distributions.

        Parameters:
        -----------
        x : float
            The x-coordinate of the point.
        y : float
            The y-coordinate of the point.
        start_distribution : scipy.stats.rv_continuous
            The probability distribution for the start date.
        end_distribution : scipy.stats.rv_continuous
            The probability distribution for the end date.
        verbose : bool, optional
            If True, prints messages about the temporal consistency check (default is False).
        """
        self.x = x
        self.y = y
        self.start_distribution = start_distribution
        self.end_distribution = end_distribution

        if verbose:
            print("Temporal consistency check...")

        # Perform checks
        self._check_distributions(verbose)

    def _check_distributions(self, verbose):
        """
        Checks the temporal consistency of the start and end distributions.

        Parameters:
        -----------
        verbose : bool
            If True, prints warnings about significant overlap or chronological inconsistencies.
        """
        overlap_ratio = self._calculate_overlap_ratio()
        if overlap_ratio > 0.25:
            print(f"Warning: Significant overlap between start and end "
                  f"distributions. Overlap ratio: {overlap_ratio:.2f}")

        start_mean = self.start_distribution.mean()
        end_mean = self.end_distribution.mean()
        if end_mean < start_mean:
            print(f"Warning: End date distribution mean ({end_mean}) is earlier "
                  f"than start date distribution mean ({start_mean}). Possible "
                  f"data error.")

    def _calculate_overlap_ratio(self):
        """
        Calculates the overlap ratio between the start and end distributions.

        Returns:
        --------
        float
            The overlap ratio between the start and end distributions.
        """
        # Define a reasonable range for integration
        range_min = min(self.start_distribution.ppf(0.01), 
                        self.end_distribution.ppf(0.01))
        range_max = max(self.start_distribution.ppf(0.99), 
                        self.end_distribution.ppf(0.99))
        
        # Generate a dense range of values for the PDFs
        x = np.linspace(range_min, range_max, 1000)
        
        # Compute the PDF values
        start_pdf = self.start_distribution.pdf(x)
        end_pdf = self.end_distribution.pdf(x)
        
        # Calculate the overlap area using the minimum of the two PDFs
        overlap_pdf = np.minimum(start_pdf, end_pdf)
        overlap_area = simps(overlap_pdf, x)
        
        # Calculate the total area of the two distributions
        total_area_start = simps(start_pdf, x)
        total_area_end = simps(end_pdf, x)
        
        # Calculate combined area
        combined_area = total_area_start + total_area_end
        
        # Calculate the overlap ratio
        if combined_area == 0:
            print(f"Warning: Sum of density integrals is zero! Check that start "
                  f"and end dates are present and, if constant, not identical.")
            overlap_ratio = np.nan
        else:
            overlap_ratio = overlap_area / combined_area
        
        return overlap_ratio

    def calculate_inclusion_probability(self, time_slice):
        """
        Calculates the inclusion probability of the point for a given time slice.

        Parameters:
        -----------
        time_slice : float
            The specific time slice to calculate the inclusion probability for.

        Returns:
        --------
        float
            The inclusion probability for the given time slice.
        """
        start_prob = self.start_distribution.cdf(time_slice)
        if start_prob <= 0:  # If start probability is zero or negative
            return 0.0
        end_prob = self.end_distribution.sf(time_slice)
        return start_prob * end_prob
    
    def _repr_distribution(self, dist):
        """
        Returns a string representation of the distribution.

        Parameters:
        -----------
        dist : scipy.stats.rv_continuous
            The distribution to represent as a string.

        Returns:
        --------
        str
            The string representation of the distribution.
        """
        if dist.dist.name == 'norm':
            loc = dist.mean()
            scale = dist.std()
            return f"norm(loc={loc}, scale={scale})"
        elif dist.dist.name == 'ddelta':
            return f"ddelta(d={dist.d})"
        else:
            params = {key: value for key, value in dist.__dict__.items() if not key.startswith('_')}
            param_str = ', '.join([f"{key}={value}" for key, value in params.items()])
            return f"{dist.dist.name}({param_str})"

    def __repr__(self):
        """
        Returns a string representation of the Point instance.

        Returns:
        --------
        str
            The string representation of the Point instance.
        """
        start_repr = self._repr_distribution(self.start_distribution)
        end_repr = self._repr_distribution(self.end_distribution)
        return (f"Point(x={self.x}, y={self.y}, "
                f"start_distribution={start_repr}, "
                f"end_distribution={end_repr})")

def in_probs(points, time_slices):
    n_points = len(points)
    n_slices = len(time_slices)
    inclusion_probs = np.zeros((n_points, n_slices))

    for i, point in enumerate(points):
        for j, t in enumerate(time_slices):
            inclusion_probs[i, j] = point.calculate_inclusion_probability(t)

    return inclusion_probs

def mc_samples(points, 
               time_slices, 
               inclusion_probs=None, 
               num_iterations=1000,
               csr=False):
    """
    Generate Monte Carlo samples for each time slice based on precomputed or 
    dynamically calculated inclusion probabilities. Supports CSR sampling.

    Parameters:
    points (list of Point objects): List of Point objects.
    time_slices (array-like): Array of time slices.
    inclusion_probs (np.ndarray, optional): Precomputed inclusion probabilities. Default is None.
    num_iterations (int): Number of Monte Carlo iterations.
    csr (bool): Whether to perform CSR sampling. Default is False.

    Returns:
    list: A list of lists where each sublist contains tuples of time slice and included points.
    """
    if inclusion_probs is None:
        # Calculate inclusion probabilities dynamically
        inclusion_probs = np.zeros((len(points), len(time_slices)))
        for i, point in enumerate(points):
            for j, t in enumerate(time_slices):
                inclusion_probs[i, j] = point.calculate_inclusion_probability(t)

    point_sets = []

    if csr:
        # Calculate bounding box and get CSR sample
        x_min, x_max, y_min, y_max = get_box(points)

    for _ in range(num_iterations):
        if csr:
            # get new point coordinates
            csr_points = csr_sample(points, x_min, x_max, y_min, y_max)
            current_points = csr_points
        else:
            current_points = points

        iteration_set = []
        for j, t in enumerate(time_slices):
            included_points = np.array([
                [point.x, point.y] 
                for point, prob in zip(current_points, inclusion_probs[:, j]) 
                if random.random() < prob
            ])
            iteration_set.append((t, included_points))
        point_sets.append(iteration_set)
    
    return point_sets

def temporal_cluster(point_sets, 
                     distances, 
                     time_slices,  
                     calc_K=True, 
                     calc_L=False, 
                     calc_G=False,
                     focal_points=None):
    """
    Calculate Ripley's K, Ripley's L, and pair correlation function over time.

    Parameters:
    point_sets (list): List of point sets for each mc iteration.
    distances (np.ndarray): Array of distances at which to calculate the functions.
    time_slices (np.ndarray): Array of time slices.
    calc_K (bool): Whether to calculate Ripley's K function. Default is True.
    calc_L (bool): Whether to calculate Ripley's L function. Default is False.
    calc_G (bool): Whether to calculate the pair correlation function. Default is False.
    focal_points (list, optional): List of focal points for each mc iteration. Default is None.

    Returns:
    tuple: A tuple containing the results for K, L, and/or G functions as 3D arrays.
    """
    if focal_points and len(point_sets) != len(focal_points):
        raise ValueError("The length of simulations must match the length of focal_points.")
    
    num_distances = len(distances)
    num_slices = len(time_slices)
    num_iterations = len(point_sets)
    
    k_results = np.zeros((num_distances, num_slices, num_iterations)) if calc_K else None
    l_results = np.zeros((num_distances, num_slices, num_iterations)) if calc_L else None
    g_results = np.zeros((num_distances, num_slices, num_iterations)) if calc_G else None

    for iteration_index, iteration_set in enumerate(point_sets):
        for t_index, (t, points) in enumerate(iteration_set):
            if len(points) < 2:
                continue
            
            coordinates = np.array(points)
            if focal_points:
                focal_coords = np.array(focal_points[iteration_index][t_index][1])
                if calc_K:
                    dists = distance.cdist(focal_coords, coordinates)
                    k_values = np.sum(dists <= distances[:, None, None], axis=1).sum(axis=-1)
                    k_results[:, t_index, iteration_index] = k_values / (len(focal_coords) * len(coordinates))
                if calc_L:
                    dists = distance.cdist(focal_coords, coordinates)
                    l_values = np.sqrt(k_values / np.pi) - distances
                    l_results[:, t_index, iteration_index] = l_values
                if calc_G:
                    dists = distance.cdist(focal_coords, coordinates)
                    k_values = np.sum(dists <= distances[:, None, None], axis=1).sum(axis=-1)
                    g_values = np.gradient(k_values, distances) / (2 * np.pi * distances)
                    g_results[:, t_index, iteration_index] = g_values
            else:
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
                      bw = 1, 
                      density = False, 
                      max_distance = None,
                      focal_points = None):
    """
    Calculate pairwise distance densities over time.

    Parameters:
    simulations (list): List of simulated point sets from mc_samples.
    time_slices (array-like): Array of time slices.
    bw (float): Bin width for histograms or bandwidth for KDE.
    density (bool): If True, use KDE; otherwise, use histograms.
    focal_points (list, optional): List of focal points for each mc iteration. Default is None.
    
    Returns:
    np.ndarray: A 3D array where the dimensions are distances (support), time slices, and iterations.
    """
    num_slices = len(time_slices)
    num_iterations = len(simulations)
    all_distances = []

    # Loop over simulations and time slices to collect pairwise distances
    if focal_points:
        for sim in range(num_iterations):
            distances_for_time_slices = []
            for time in range(num_slices):
                t, pts = simulations[sim][time]
                _, f_pts = focal_points[sim][time]
                if (len(pts) < 1) or (len(f_pts) < 1):
                    distances_for_time_slices.append(None)  # Use None to indicate no distances
                    continue
                pairwise_distances = distance.cdist(f_pts, pts)
                distances_for_time_slices.append(pairwise_distances)
            all_distances.append(distances_for_time_slices)

    else:
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
    points (list of Point): List of original Point objects.
    x_min (float): Minimum x coordinate for the randomization.
    x_max (float): Maximum x coordinate for the randomization.
    y_min (float): Minimum y coordinate for the randomization.
    y_max (float): Maximum y coordinate for the randomization.
    
    Returns:
    list of Point: List of CSR sampled Point objects with randomized locations.
    """
    randomized_points = []

    for point in points:
        new_x = np.random.uniform(x_min, x_max)
        new_y = np.random.uniform(y_min, y_max)
        randomized_point = Point(new_x, new_y, point.start_distribution, point.end_distribution)
        randomized_points.append(randomized_point)

    return randomized_points

def p_diff(pairwise_density, csr_pairwise_density, condition='greater'):
    """
    Calculate the probability that the density difference meets a specified condition,
    propagating chronological uncertainty from MC iterations.

    Parameters:
    pairwise_density (np.ndarray): Array of pairwise densities with shape (distances, time_slices, iterations).
    csr_pairwise_density (np.ndarray): Array of CSR pairwise densities with shape (distances, time_slices, iterations).
    condition (str): Condition to apply for probability calculation ('greater' or 'less').

    Returns:
    np.ndarray: Probability values with shape (distances, time_slices).
    """
    # Calculate differences for each iteration
    density_diff = pairwise_density - csr_pairwise_density

    # Calculate mean and standard deviation of the differences
    mean_density_diff = np.mean(density_diff, axis=2)
    std_density_diff = np.std(density_diff, axis=2)
    
    # Avoid division by zero in case of very small std deviation
    epsilon = np.finfo(float).eps
    std_density_diff[std_density_diff < epsilon] = epsilon
    
    # Calculate z-scores
    z_scores = mean_density_diff / std_density_diff
    
    if condition == 'greater':
        # Calculate probability for p > 0
        p_values = 1 - norm.cdf(z_scores)
    elif condition == 'less':
        # Calculate probability for p < 0
        p_values = norm.cdf(z_scores)
    else:
        raise ValueError("Invalid condition. Use 'greater' or 'less'.")
    
    return p_values