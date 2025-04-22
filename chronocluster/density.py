#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Christopher Carleton & Shuang Song
# @Contact   : carleton@gea.mpg.de
# GitHub   : https://github.com/wccarleton/chronocluster_dark

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor as pt
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from skimage.feature import peak_local_max
from sklearn.mixture import GaussianMixture

from chronocluster.data.simdata import synth_sample
from chronocluster.utils import get_box


def kde_time(
    points,
    time_slice,
    bandwidth,
    grid,
    output_shape=None,
    kde_method=gaussian_kde,
    *args,
    **kwargs,
):
    """
    Computes a KDE for a given time slice, incorporating the inclusion
    probabilities of each point as weights. Allows for flexibility in KDE
    calculation by passing in different KDE methods.

    Parameters:
    -----------
    points : list of Point
        List of Point objects, each with spatial coordinates and temporal distributions.
    time_slice : float
        The time slice for which to calculate the KDE.
    bandwidth : float
        Bandwidth for the KDE given in data units (will be the SD of the Gaussian kernel)
    grid : np.ndarray
        A 2D array of spatial coordinates for KDE evaluation.
    kde_method : callable, optional
        A method for calculating a KDE with the following signature:
        kde_method(dataset : array_like, bw_method : scalar or callable, weights : array_like, *args, **kwargs) -> callable
        This function should return a callable that can be used to evaluate the KDE
        over a given grid of points. Defaults to `scipy.stats.gaussian_kde`.
    *args : tuple
        Additional positional arguments passed to `kde_method`.
    **kwargs : dict
        Additional keyword arguments passed to `kde_method`.

    Returns:
    --------
    kde_values : np.ndarray
        KDE values evaluated over the grid.
    """
    # Get inclusion probabilities for the given time slice
    inclusion_probs = np.array(
        [point.calculate_inclusion_probability(time_slice) for point in points]
    )

    # Extract spatial coordinates
    x_coords = np.array([point.x for point in points])
    y_coords = np.array([point.y for point in points])

    # Stack coordinates for KDE input
    coordinates = np.vstack([x_coords, y_coords])

    # Initialize KDE with weights using the specified method
    kde = kde_method(
        coordinates, bw=bandwidth, weights=inclusion_probs, *args, **kwargs
    )

    # Evaluate KDE on the grid
    kde_values = kde(grid.T)  # Transpose grid for compatibility

    # reshape if necessary
    if output_shape is None:
        return kde_values
    else:
        return kde_values.reshape(output_shape)


def custom_kde(dataset, bw, weights=None):
    """
    Custom KDE function compatible with kde_time, following the interface of gaussian_kde.
    This function returns a callable that computes the KDE with a fixed Gaussian bandwidth
    in data units.

    Parameters:
    -----------
    dataset : np.ndarray
        A 2D array with shape (2, n_points) or (n_points, 2) containing the spatial coordinates
        of data points. If shape is (2, n_points), it will be transposed internally.
    bw : float
        The desired bandwidth in data units (e.g., meters for UTM coordinates).
    weights : np.ndarray, optional
        An array of weights for each point in `dataset`. Defaults to uniform weights if None.

    Returns:
    --------
    callable
        A function that takes a grid of points (in any shape) and evaluates the KDE
        over that grid, reshaping the result to match the grid's original shape.
    """
    # Transpose dataset if it has shape (2, n_points)
    if dataset.shape[0] == 2 and dataset.ndim == 2:
        dataset = dataset.T

    # Ensure dataset is a 2D array with two columns
    if dataset.ndim != 2 or dataset.shape[1] != 2:
        raise ValueError("Dataset must be a 2D array with shape (n_points, 2)")

    # Set default weights if none are provided
    if weights is None:
        weights = np.ones(dataset.shape[0])

    # Define the callable function for KDE evaluation
    def evaluate(grid):
        # Calculate squared distances between grid points and data points
        distances = cdist(grid.T, dataset, metric="sqeuclidean")

        # Apply the Gaussian kernel
        kde_values = np.dot(np.exp(-distances / (2 * bw**2)), weights)

        # Normalize by bandwidth and total weights to get density estimates
        kde_values /= (2 * np.pi * bw**2) * np.sum(weights)

        return kde_values

    return evaluate


def kde_peaks(
    kde_values=None,
    x_mesh=None,
    y_mesh=None,
    points=None,
    num_peaks=5,
    peak_finder=find_peaks,
    *args,
    **kwargs,
):
    """
    Identifies peaks in the KDE surface or point distribution using a specified peak-finding algorithm.

    Parameters:
    -----------
    kde_values : np.ndarray, optional
        2D array of KDE values over the grid, directly from kde_time. Only required for `find_peaks` and `peak_local_max`.
    x_mesh, y_mesh : np.ndarray, optional
        Mesh grids of x and y coordinates used to evaluate kde_values. Only required for `find_peaks` and `peak_local_max`.
    points : list of Point, optional
        List of Point objects with x and y coordinates, used when peak_finder is `gmm_peak_finder` or `pymc_gmm_peak_finder`.
    num_peaks : int
        Maximum number of peaks to identify on the KDE surface or point distribution.
    peak_finder : callable
        Peak-finding function to use. Options include `scipy.signal.find_peaks`, `peak_local_max`, `gmm_peak_finder`, or `pymc_gmm_peak_finder`.
    *args, **kwargs :
        Additional arguments passed to the peak-finding function.

    Returns:
    --------
    peaks : np.ndarray
        Array of shape (num_peaks, 2) containing the x, y coordinates of the identified peaks.
    weights : np.ndarray or None
        Array of sorted weights corresponding to each peak, in descending order, if applicable.
    """
    if peak_finder == find_peaks:
        if kde_values is None or x_mesh is None or y_mesh is None:
            raise ValueError(
                "kde_values, x_mesh, and y_mesh must be provided for find_peaks."
            )

        # Use scipy's find_peaks on the flattened KDE surface
        peak_indices = peak_finder(kde_values.ravel(), *args, **kwargs)[0]
        peak_coords = np.column_stack(
            (x_mesh.ravel()[peak_indices], y_mesh.ravel()[peak_indices])
        )
        weights = None  # No weights returned by find_peaks

    elif peak_finder == peak_local_max:
        if kde_values is None or x_mesh is None or y_mesh is None:
            raise ValueError(
                "kde_values, x_mesh, and y_mesh must be provided for peak_local_max."
            )

        # Use skimage's peak_local_max for local maxima detection on the KDE surface
        peak_indices = peak_finder(kde_values, *args, **kwargs)
        peak_coords = np.column_stack(
            (
                x_mesh[peak_indices[:, 0], peak_indices[:, 1]],
                y_mesh[peak_indices[:, 0], y_mesh[peak_indices[:, 1]]],
            )
        )
        weights = None  # No weights returned by peak_local_max

    elif peak_finder == gmm_peak_finder:
        # Ensure points are provided for GMM peak finding
        if points is None:
            raise ValueError(
                "The 'points' parameter must be provided when using gmm_peak_finder."
            )

        # Use GMM to identify peaks based on point distribution
        ranked_peaks, ranked_weights = gmm_peak_finder(
            points, num_components=num_peaks, *args, **kwargs
        )
        peak_coords = ranked_peaks
        weights = ranked_weights

    elif peak_finder == pymc_gmm_peak_finder:
        # Ensure points are provided for PyMC GMM peak finding
        if points is None:
            raise ValueError(
                "The 'points' parameter must be provided when using pymc_gmm_peak_finder."
            )

        # Check for required parameters in kwargs
        required_params = ["time_slice", "target_scale", "target_scale_sd"]
        for param in required_params:
            if param not in kwargs:
                raise ValueError(
                    f"Missing required parameter '{param}' for pymc_gmm_peak_finder."
                )

        # Extract PyMC-specific parameters without defaults
        time_slice = kwargs.pop("time_slice")
        target_scale = kwargs.pop("target_scale")
        target_scale_sd = kwargs.pop("target_scale_sd")

        # Optional parameters with defaults
        n_samples = kwargs.pop("n_samples", 1000)
        num_components = kwargs.pop("num_components", num_peaks)
        w_threshold = kwargs.pop("w_threshold", 1 / num_components)

        # Use the custom PyMC GMM model to identify peaks based on point distribution
        peak_coords, weights, trace = pymc_gmm_peak_finder(
            points,
            time_slice=time_slice,
            n_samples=n_samples,
            num_components=num_components,
            target_scale=target_scale,
            target_scale_sd=target_scale_sd,
            w_threshold=w_threshold,
            **kwargs,
        )
    else:
        raise ValueError(
            "Unsupported peak-finding function. Please use find_peaks, peak_local_max, gmm_peak_finder, or pymc_gmm_peak_finder."
        )

    # Sort peaks by density or weight and select the top `num_peaks`
    if weights is None:
        # Sort by KDE values for non-GMM methods
        sorted_peak_coords = peak_coords[
            np.argsort(kde_values.ravel()[peak_indices])[::-1]
        ]
        peaks = sorted_peak_coords[:num_peaks]
        weights = None
    else:
        # For GMM methods, peaks are already sorted by weight in the peak finder
        peaks = peak_coords[:num_peaks]
        weights = weights[:num_peaks]

    # return
    if peak_finder == pymc_gmm_peak_finder:
        # return the trace as well
        return peaks, weights, trace
    else:
        return peaks, weights


def peak_local_max_finder(kde_values, x_mesh, y_mesh, min_distance=5, num_peaks=10):
    """
    Uses the `peak_local_max` function from `skimage.feature` to identify peaks in a 2D KDE surface.

    Parameters:
    -----------
    kde_values : np.ndarray
        2D array of KDE values over the grid.
    x_mesh, y_mesh : np.ndarray
        Mesh grids of x and y coordinates used to evaluate kde_values.
    min_distance : int, optional
        Minimum number of pixels separating peaks in a region of `2 * min_distance + 1` (i.e., a peak is
        only detected if it has the maximum value within its local region).
    num_peaks : int, optional
        Maximum number of peaks to return.

    Returns:
    --------
    peak_coords : np.ndarray
        Array of shape (num_peaks, 2) containing the x, y coordinates of the identified peaks.
    """
    # Use `peak_local_max` to identify local maxima in the KDE surface
    peak_indices = peak_local_max(
        kde_values, min_distance=min_distance, num_peaks=num_peaks
    )

    # Get x, y coordinates for the identified peaks
    peak_coords = np.column_stack(
        (
            x_mesh[peak_indices[:, 0], peak_indices[:, 1]],
            y_mesh[peak_indices[:, 0], peak_indices[:, 1]],
        )
    )

    return peak_coords


def gmm_peak_finder(points, num_components=8):
    """
    Uses a Gaussian Mixture Model (GMM) to identify peaks in a point distribution.

    Parameters:
    -----------
    points : list of Point
        List of Point objects, each with x and y attributes.
    num_components : int, optional
        Number of GMM components (peaks) to fit. Default is 8.

    Returns:
    --------
    ranked_peaks : np.ndarray
        Array of shape (num_components, 2) containing the x, y coordinates of the identified peaks.
    ranked_weights : np.ndarray
        Array of weights for each peak, in descending order of importance.
    """
    # Extract coordinates from points
    coordinates = np.array([[point.x, point.y] for point in points])

    # Fit the GMM to the point coordinates
    gmm = GaussianMixture(n_components=num_components, covariance_type="diag")
    gmm.fit(coordinates)

    # Extract the means (coordinates of peaks) and weights
    peak_coordinates = gmm.means_
    peak_weights = gmm.weights_

    # Rank-order the peaks by their weights
    ranked_indices = np.argsort(peak_weights)[::-1]  # Sort in descending order
    ranked_peaks = peak_coordinates[ranked_indices]
    ranked_weights = peak_weights[ranked_indices]

    return ranked_peaks, ranked_weights


def pymc_gmm_peak_finder(
    points,
    time_slice,
    n_samples,
    num_components,
    target_scale,
    target_scale_sd,
    w_threshold=None,
    sampler="MH",
    **kwargs,
):
    """
    Identifies peaks using a Bayesian GMM implemented in PyMC, with bounding box derived from data.

    Parameters:
    -----------
    points : list of Point
        List of Point objects with x and y coordinates.
    time_slice : int
        The specific time slice for which to calculate the KDE and sample points.
    n_samples : int, optional
        Number of synthetic samples to generate for weighting.
    num_components : int, optional
        Maximum number of components (peaks) for the GMM.
    target_scale : float, optional
        Centered scale for the covariance of the components.
    target_scale_sd : float, optional
        Standard deviation around the target scale.
    w_threshold : float, optional
        Threshold for component weights to be considered as significant peaks.
    sampler : str, optional
        Select the sampler used for the PyMC model (e.g., 'MH' for
        Metropolis-Hastings or 'NUTS' for No-U-Turn Sampler). Deafults to 'MH'.
    **kwargs : dict
        Additional parameters for PyMC's pm.sample function (e.g., draws, chains, tune).

    Returns:
    --------
    peak_coords : np.ndarray
        Array of shape (num_peaks, 2) containing the x, y coordinates of the identified peaks.
    weights : np.ndarray
        Array of weights corresponding to each identified peak.
    """
    # Generate weighted sample based on inclusion probabilities for the time slice
    synth_points = synth_sample(points, n_samples=n_samples, time_slice=time_slice)
    coordinates = np.array([[point.x, point.y] for point in synth_points])

    # Calculate bounding box variance from points
    min_x, min_y, max_x, max_y = get_box(points)
    bounding_box_variance = (
        (max_x - min_x) / 2
    ) ** 2  # Variance to cover half the bounding box width

    with pm.Model():
        # Dirichlet Process for mixture weights
        w = pm.Dirichlet("w", a=np.ones(num_components))

        # Means for each component
        data_mean = np.mean(coordinates, axis=0)
        means = pm.MvNormal(
            "means",
            mu=data_mean,
            cov=np.eye(2) * bounding_box_variance,
            shape=(num_components, 2),
        )

        # Covariance matrices for each component
        chol_factors = []
        for i in range(num_components):
            # Use `.dist()` for unregistered distributions
            sd_scale_i = pm.TruncatedNormal.dist(
                mu=target_scale, sigma=target_scale_sd, lower=0
            )

            # Define Cholesky decomposition
            chol_i, _, _ = pm.LKJCholeskyCov(
                f"chol_{i}",  # Unique name for each Cholesky factor
                n=2,
                eta=2,
                sd_dist=sd_scale_i,
                compute_corr=True,
            )
            chol_factors.append(chol_i)

        # Stack Cholesky factors using `pt.stacklists`
        chol_factors_stacked = pm.Deterministic(
            "chol_factors", pt.tensor.stack(chol_factors, axis=0)
        )

        # Multivariate Normal for each component
        components = [
            pm.MvNormal.dist(mu=means[i], chol=chol_factors_stacked[i])
            for i in range(num_components)
        ]

        # Mixture likelihood for observed data
        pm.Mixture("gmm", w=w, comp_dists=components, observed=coordinates)

        # Track "importance" of each component relative to uniform prior
        w_threshold = w_threshold or (1 / num_components)
        pm.Deterministic("importance", w - w_threshold)

        # Sampling parameters
        draws = kwargs.pop("draws", 4000)
        chains = kwargs.pop("chains", 1)
        tune = kwargs.pop("tune", 3000)

        # Select the sampling method
        if sampler == "MH":
            step = pm.Metropolis()
        elif sampler == "NUTS":
            step = pm.NUTS()
        else:
            raise ValueError(
                "Unsupported sampler. Choose 'MH' for Metropolis-Hastings or 'NUTS'."
            )

        # Run MCMC sampling
        trace = pm.sample(draws=draws, chains=chains, step=step, tune=tune, **kwargs)

    # Extract posterior means of component weights and coordinates
    posterior_weights = trace.posterior["w"].mean(dim=("chain", "draw")).values
    posterior_means = trace.posterior["means"].mean(dim=("chain", "draw")).values

    # Filter peaks by weight threshold, if provided
    valid_indices = np.where(posterior_weights > w_threshold)[0]
    peak_coords = posterior_means[valid_indices]
    weights = posterior_weights[valid_indices]

    # Sort by weight for output
    sorted_indices = np.argsort(weights)[::-1]
    peak_coords = peak_coords[sorted_indices]
    weights = weights[sorted_indices]

    return peak_coords, weights, trace


def rank_peaks(trace, significance=0.95, source_param="w"):
    """
    Summarizes and ranks the importance of GMM components based on posterior weights and coordinates.

    Parameters:
    -----------
    trace : arviz.InferenceData
        The MCMC trace output from PyMC.
    significance : float
        The desired credible interval width for HDI (e.g., 0.95 for a 95% HDI).
    source_param : str
        Specifies whether to rank peaks based on 'w' (weights) or 'importance' (i.e., w_i - w_threshold).

    Returns:
    --------
    summary_df : pd.DataFrame
        DataFrame containing ranked component weights or importance, coordinates, covariance matrices, and HDI for the selected metric.
    """
    # Select the source for ranking: 'w' or 'importance'
    if source_param == "w":
        # Compute the mean of weights across chains and draws
        posterior_values = trace.posterior["w"].mean(dim=("chain", "draw")).values
    elif source_param == "importance":
        # Compute the mean of importance across chains and draws
        posterior_values = (
            trace.posterior["importance"].mean(dim=("chain", "draw")).values
        )
    else:
        raise ValueError("Invalid source_param. Choose either 'w' or 'importance'.")

    # Compute the mean of component means across chains and draws
    posterior_means = trace.posterior["means"].mean(dim=("chain", "draw")).values

    # Extract Cholesky factors for each component
    chol_factors = (
        trace.posterior["chol_factors"].mean(dim=("chain", "draw")).values
    )  # Shape: (num_components, 2, 2)

    # Compute covariance matrices for each component
    covariances = []
    for chol_matrix in chol_factors:
        # Compute the covariance matrix: Σ = L @ L.T
        cov_matrix = chol_matrix @ chol_matrix.T
        covariances.append(cov_matrix)

    # Sort components by selected metric in descending order
    sorted_indices = np.argsort(posterior_values)[::-1]
    sorted_values = posterior_values[sorted_indices]
    sorted_means = posterior_means[sorted_indices]
    sorted_covariances = [covariances[i] for i in sorted_indices]

    # Calculate HDI for the selected source parameter
    hdi_values = az.hdi(trace, var_names=[source_param], hdi_prob=significance)[
        source_param
    ].values

    # Reorder HDI values based on sorted component indices and format as tuples
    sorted_hdi_values = [tuple(hdi) for hdi in hdi_values[sorted_indices]]

    # Prepare summary table
    summary_data = {
        "Rank": np.arange(1, len(sorted_values) + 1),
        source_param.capitalize(): sorted_values,
        "Coordinates": [
            tuple(mean) for mean in sorted_means
        ],  # Store coordinates as tuples
        "Covariances": sorted_covariances,  # Store covariance matrices for each component
        f"{int(significance * 100)}% HDI ({source_param.capitalize()})": sorted_hdi_values,  # Store HDI as tuples
    }
    summary_df = pd.DataFrame(summary_data)

    return summary_df


def radial_kde(
    points, time_slice, bandwidth, center, radii, ring_sample_density, *args, **kwargs
):
    """
    Computes a radial density profile by evaluating a KDE at points along rings of increasing radii.

    Parameters:
    -----------
    points : list of Point
        List of Point objects, each with spatial coordinates and temporal distributions.
    time_slice : float
        The time slice for which to calculate the KDE.
    bandwidth : float
        Bandwidth for the KDE.
    center : tuple
        The (x, y) coordinates of the center point from which to measure radii.
    radii : array_like
        Array of radii at which to evaluate the KDE.
    ring_sample_density : float
        Density of points per unit length of each ring.
    *args : tuple
        Additional positional arguments passed to `kde_time_func`.
    **kwargs : dict
        Additional keyword arguments passed to `kde_time_func`.

    Returns:
    --------
    radial_density : np.ndarray
        Array of KDE values averaged over points along each radius, representing the radial density profile.
    """
    radial_density = []

    # Generate points on rings at each radius
    for r in radii:
        # Calculate the number of points for the current ring based on its circumference
        circumference = 2 * np.pi * r
        num_points = max(
            1, int(circumference * ring_sample_density)
        )  # At least 1 point per ring

        # Generate points evenly spaced around the ring
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        x_ring = center[0] + r * np.cos(angles)
        y_ring = center[1] + r * np.sin(angles)
        ring_points = np.vstack([x_ring, y_ring]).T  # Shape (num_points, 2)

        # Evaluate KDE at the points on the ring
        kde_values = kde_time(
            points, time_slice, bandwidth, ring_points, *args, **kwargs
        )

        # Average KDE values around the ring to get a single density value for this radius
        radial_density.append(np.mean(kde_values))

    return np.array(radial_density)


def radial_density(coordinates, centers, radial_steps=10, max_radius=5000):
    """
    Calculates radial density profiles for a given set of centers.

    Parameters:
    -----------
    coordinates : np.ndarray
        Array of shape (n_points, 2) containing the x, y coordinates of the data points.
    centers : np.ndarray
        Array of shape (num_centers, 2) containing the x, y coordinates of the KDE peaks.
    radial_steps : int
        Number of radial steps to calculate the density.
    max_radius : float
        Maximum radius for the radial density profile.

    Returns:
    --------
    radial_profiles : dict
        Dictionary with center indices as keys and lists of density values at each radius as values.
    radii : np.ndarray
        Array of radius values corresponding to each radial step.
    """
    radii = np.linspace(0, max_radius, radial_steps)
    radial_profiles = {i: [] for i in range(len(centers))}

    for i, center in enumerate(centers):
        for radius in radii:
            # Find points within the current radius
            distances = cdist([center], coordinates)[0]
            points_in_radius = coordinates[distances <= radius]
            area = np.pi * radius**2  # Area of the circle at this radius
            density = len(points_in_radius) / area if area > 0 else 0
            radial_profiles[i].append(density)

    return radial_profiles, radii
