#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Christopher Carleton & Shuang Song
# @Contact   : carleton@gea.mpg.de
# GitHub   : https://github.com/wccarleton/chronocluster

import random

import numpy as np
import pointpats.distance_statistics as dstats
from scipy.spatial import distance
from scipy.stats import gaussian_kde, norm

from chronocluster.point import Point


def in_probs(points, time_slices):
    """
    Calculate inclusion probabilities for multiple points and time slices.

    Parameters
    -----------
    points : list[Point]
        List of Point objects.
    time_slices : array-like
        Array of time slices.

    Returns
    --------
    np.ndarray
        2D array of inclusion probabilities, shape (n_points, n_slices).
    """
    time_slices_array = np.asarray(time_slices)
    return np.vstack(
        [point.calculate_inclusion_probability(time_slices_array) for point in points]
    )


def mc_samples(
    points,
    time_slices,
    inclusion_probs=None,
    num_iterations=1000,
    null_model=None,
    **kwargs,
):
    """
    Generate Monte Carlo samples for each time slice based on precomputed or
    dynamically calculated inclusion probabilities. Allows customizable null models.

    Parameters:
    -----------
    points : list of Point objects
        List of Point objects representing the observed dataset.
    time_slices : array-like
        Array of time slices.
    inclusion_probs : np.ndarray, optional
        Precomputed inclusion probabilities (shape: [n_points, n_time_slices]).
        If None, inclusion probabilities are calculated dynamically.
    num_iterations : int
        Number of Monte Carlo iterations.
    null_model : callable, optional
        A function for generating null realizations. Must take `points` as the
        first argument and return either:
        - A list of Point objects (spatial realization).
        - A tuple where the first element is the list of Point objects.
    **kwargs : dict
        Additional keyword arguments to pass to the `null_model` callable.

    Returns:
    --------
    list of lists :
        A list of sublists where each sublist contains tuples of time slice
        and included points for one realization.
    """
    # Calculate inclusion probabilities if not provided
    if inclusion_probs is None:
        inclusion_probs = in_probs(points, time_slices)

    point_sets = []

    # Generate Monte Carlo realizations
    for _ in range(num_iterations):
        # Apply the null model if provided, otherwise use the original points
        if null_model:
            result = null_model(points, **kwargs)
            # Check if result is a tuple or single object
            if isinstance(result, tuple):
                spatial_realization = result[0]  # Use the first element
            else:
                spatial_realization = (
                    result  # Use the result directly if it's a single object
                )
        else:
            spatial_realization = points  # Use original points directly

        # Sample time for the spatial realization
        iteration_set = []
        for j, t in enumerate(time_slices):
            included_points = np.array(
                [
                    [point.x, point.y]
                    for point, prob in zip(spatial_realization, inclusion_probs[:, j])
                    if random.random() < prob
                ]
            )
            iteration_set.append((t, included_points))
        point_sets.append(iteration_set)

    return point_sets


def temporal_cluster(
    point_sets,
    distances,
    time_slices,
    calc_K=True,
    calc_L=False,
    calc_G=False,
    focal_points=None,
):
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
    num_distances = len(distances)
    num_slices = len(time_slices)
    num_iterations = len(point_sets)

    k_results = (
        np.zeros((num_distances, num_slices, num_iterations)) if calc_K else None
    )
    l_results = (
        np.zeros((num_distances, num_slices, num_iterations)) if calc_L else None
    )
    g_results = (
        np.zeros((num_distances, num_slices, num_iterations)) if calc_G else None
    )

    if focal_points:
        for sim in range(num_iterations):
            for t in range(num_slices):
                _, points = point_sets[sim][t]
                _, focal_pts = focal_points[sim][t]
                if (len(points) < 1) or (len(focal_pts) < 1):
                    continue

                coords = np.array(points)
                focal_coords = np.array(focal_pts)

                if calc_K:
                    dists = distance.cdist(focal_coords, coords)
                    k_values = np.sum(dists <= distances[:, None, None], axis=1).sum(
                        axis=-1
                    )
                    k_results[:, t, sim] = k_values / (len(focal_coords) * len(coords))
                if calc_L:
                    l_values = np.sqrt(k_results[:, t, sim] / np.pi) - distances
                    l_results[:, t, sim] = l_values
                if calc_G:
                    dists = distance.cdist(focal_coords, coords)
                    k_values = np.sum(dists <= distances[:, None, None], axis=1).sum(
                        axis=-1
                    )
                    g_values = np.gradient(k_values, distances) / (
                        2 * np.pi * distances
                    )
                    g_results[:, t, sim] = g_values
    else:
        for sim in range(num_iterations):
            for t in range(num_slices):
                _, points = point_sets[sim][t]
                if len(points) < 2:
                    continue

                coords = np.array(points)
                if calc_K:
                    support, k_estimate = dstats.k(coords, support=distances)
                    k_results[:, t, sim] = k_estimate
                if calc_L:
                    support, l_estimate = dstats.l(coords, support=distances)
                    l_results[:, t, sim] = l_estimate
                if calc_G:
                    support, k_estimate = dstats.k(coords, support=distances)
                    g_estimate = np.gradient(k_estimate, distances) / (
                        2 * np.pi * distances
                    )
                    g_results[:, t, sim] = g_estimate

    return k_results, l_results, g_results


def temporal_pairwise(
    point_sets,
    time_slices,
    bw=1,
    use_kde=False,
    kde_sample_n=100,
    max_distance=None,
    focal_points=None,
    kde_custom=None,
    **kwargs,
):
    """
    Calculate pairwise distance densities over time.

    Parameters:
    -----------
    point_sets : list
        List of point sets from mc_samples.
    time_slices : array-like
        Array of time slices.
    bw : float
        Bandwidth for KDE or bin width for histograms in data units.
    use_kde : bool
        If True, use KDE; otherwise, use histograms.
    kde_sample_n : int
        Number of equally spaced samples for the KDE function.
    max_distance : float, optional
        Maximum distance for binning/KDE support. Calculated from data if None.
    focal_points : list, optional
        List of focal points for each iteration. Defaults to None.
    kde_custom : callable, optional
        A custom KDE callable. If None, defaults to scipy's gaussian_kde.
    kwargs : additional arguments
        Arguments passed to the KDE callable.

    Returns:
    --------
    pairwise_density : np.ndarray
        3D array where the dimensions are distances, time slices, and iterations.
    support : np.ndarray
        Array of distance values for the KDE or histogram.
    """
    num_slices = len(time_slices)
    num_iterations = len(point_sets)
    all_distances = []

    # Collect pairwise distances
    if focal_points:
        for sim in range(num_iterations):
            distances_for_time_slices = []
            for time in range(num_slices):
                _, pts = point_sets[sim][time]
                _, f_pts = focal_points[sim][time]
                if len(pts) < 1 or len(f_pts) < 1:
                    distances_for_time_slices.append(None)
                    continue
                pairwise_distances = distance.cdist(f_pts, pts)
                if max_distance is not None:
                    pairwise_distances = pairwise_distances[pairwise_distances <= max_distance]
                if pairwise_distances.size == 0:
                    distances_for_time_slices.append(None)
                else:
                    distances_for_time_slices.append(pairwise_distances)
            all_distances.append(distances_for_time_slices)
    else:
        for simulation in point_sets:
            distances_for_time_slices = []
            for _, points in simulation:
                if len(points) < 2:
                    distances_for_time_slices.append(None)
                    continue
                pairwise_distances = distance.pdist(points)
                if max_distance is not None:
                    pairwise_distances = pairwise_distances[pairwise_distances <= max_distance]
                if pairwise_distances.size == 0:
                    distances_for_time_slices.append(None)
                else:
                    distances_for_time_slices.append(pairwise_distances)
            all_distances.append(distances_for_time_slices)

    # Calculate max_distance if not provided
    if max_distance is None:
        flat_distances = [
            dist for sublist in all_distances for dist in sublist if dist is not None
        ]
        if not flat_distances:
            raise ValueError("No valid distances found.")
        max_distance = np.max([np.max(d) for d in flat_distances])

    # Create KDE support or bins
    if use_kde:
        support = np.linspace(0, max_distance, kde_sample_n)
    else:
        bins = np.arange(0, max_distance + bw, bw)
        support = bins[:-1] + (bw / 2)

    # Initialize pairwise density array
    pairwise_density = np.zeros((len(support), num_slices, num_iterations))

    # Calculate densities
    for iteration_index in range(num_iterations):
        for t_index in range(num_slices):
            distances_for_slice = all_distances[iteration_index][t_index]
            if distances_for_slice is None:
                pairwise_density[:, t_index, iteration_index] = np.nan
                continue
            if use_kde:
                distances = np.ravel(distances_for_slice)
                if kde_custom:
                    # Use the custom KDE method
                    kde = kde_custom(
                        distances, bandwidth=bw, **kwargs
                    )  # distances should be passed as 1D
                    pairwise_density[:, t_index, iteration_index] = kde(support)
                else:
                    # Default to scipy's gaussian_kde
                    bw_scaling_factor = (0.5 * bw) / np.std(distances)
                    kde = gaussian_kde(distances, bw_method=bw_scaling_factor)
                    pairwise_density[:, t_index, iteration_index] = kde(support)
            else:
                hist, _ = np.histogram(distances_for_slice, bins=bins, density=True)
                pairwise_density[:, t_index, iteration_index] = hist

    return pairwise_density, support


def csr_sample(points, x_min, x_max, y_min, y_max):
    """
    Generate a CSR sample by randomizing the locations of the points while keeping their temporal information intact.

    Parameters:
    -----------
    points : list of Point
        List of original Point objects.
    x_min : float
        Minimum x coordinate for the randomization.
    x_max : float
        Maximum x coordinate for the randomization.
    y_min : float
        Minimum y coordinate for the randomization.
    y_max : float
        Maximum y coordinate for the randomization.

    Returns:
    --------
    tuple:
        - list of Point: CSR sampled Point objects with randomized locations.
        - dict: Metadata about the CSR realization (e.g., bounding box information).
    """
    randomized_points = []

    for point in points:
        new_x = np.random.uniform(x_min, x_max)
        new_y = np.random.uniform(y_min, y_max)
        randomized_point = Point(
            new_x, new_y, point.start_distribution, point.end_distribution
        )
        randomized_points.append(randomized_point)

    # Include metadata (e.g., bounding box used for randomization)
    metadata = {"bounding_box": (x_min, x_max, y_min, y_max)}

    return randomized_points, metadata


def bise(points):
    """
    Generates a null distribution of spatial coordinates based on the mean and covariance
    matrix of the provided Point objects and returns a list of new Point objects with
    coordinates sampled from the null distribution, along with metadata.

    Parameters:
    -----------
    points : list of Point
        List of Point objects with x, y coordinates.

    Returns:
    --------
    tuple:
        - simulated_points : list of Point
            List of new Point objects with coordinates from the simulated null distribution.
        - metadata : dict
            Dictionary containing 'mean_location', 'cov_matrix', and 'simulated_coords'.
    """
    # Extract coordinates from Point objects
    coordinates = np.array([[point.x, point.y] for point in points])

    # Step 1: Calculate mean and covariance of the observed coordinates
    mean_location = np.mean(coordinates, axis=0)
    cov_matrix = np.cov(coordinates, rowvar=False)

    # Step 2: Generate simulated coordinates from a bivariate normal distribution
    nsim = len(points)
    simulated_coords = np.random.multivariate_normal(mean_location, cov_matrix, nsim)

    # Step 3: Create new Point objects with simulated coordinates
    simulated_points = [
        Point(
            x=simulated_coords[i, 0],
            y=simulated_coords[i, 1],
            start_distribution=point.start_distribution,
            end_distribution=point.end_distribution,
        )
        for i, point in enumerate(points)
    ]

    # Prepare metadata
    metadata = {
        "mean_location": mean_location,
        "cov_matrix": cov_matrix,
        "simulated_coords": simulated_coords,
    }

    return simulated_points, metadata


def p_diff(pairwise_density, null_pairwise_density, condition="greater"):
    """
    Calculate the probability that the density difference meets a specified condition,
    propagating chronological uncertainty from MC iterations.

    Parameters:
    pairwise_density (np.ndarray): Array of pairwise densities with shape (distances, time_slices, iterations).
    null_pairwise_density (np.ndarray): Array of null hyopthesis (e.g., CSR) pairwise densities with shape (distances, time_slices, iterations).
    condition (str): Condition to apply for probability calculation ('greater' or 'less').

    Returns:
    np.ndarray: Probability values with shape (distances, time_slices).
    """
    # Calculate differences for each iteration
    density_diff = pairwise_density - null_pairwise_density

    # Calculate mean and standard deviation of the differences
    mean_density_diff = np.mean(density_diff, axis=2)
    std_density_diff = np.std(density_diff, axis=2)

    # Avoid division by zero in case of very small std deviation
    epsilon = np.finfo(float).eps
    std_density_diff[std_density_diff < epsilon] = epsilon

    # Calculate z-scores
    z_scores = mean_density_diff / std_density_diff

    if condition == "greater":
        # Calculate probability for p(Z > z)
        p_values = 1 - norm.cdf(z_scores)
    elif condition == "less":
        # Calculate probability for p(Z <= z)
        p_values = norm.cdf(z_scores)
    else:
        raise ValueError("Invalid condition. Use 'greater' or 'less'.")

    return p_values, density_diff
