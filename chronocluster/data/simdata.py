import numpy as np
import pandas as pd
from chronocluster.data.dataio import df_to_pts
from chronocluster.distributions import calrcarbon
from chronocluster.calcurves import calibration_curves

def generate_random_points(n_points, 
                           cluster_centers, 
                           cluster_stds, 
                           start_type, 
                           start_hyperparams, 
                           end_type, 
                           end_hyperparams):
    """
    Generate random points with specified cluster centers, standard deviations, and temporal distributions.

    Parameters:
    n_points (int): Number of points to generate.
    cluster_centers (list of tuples): List of (x, y) tuples representing the centers of clusters.
    cluster_stds (list of floats): List of standard deviations for each cluster.
    start_type (str): Type of the start distribution ('norm', 'uniform', 'constant', 'calrcarbon').
    start_hyperparams (list): Hyperparameters for the start distribution.
    end_type (str): Type of the end distribution ('norm', 'uniform', 'constant', 'calrcarbon').
    end_hyperparams (list): Hyperparameters for the end distribution.
    calcurve_name (str, optional): Name of the calibration curve data required for 'calrcarbon' distribution.

    Returns:
    list of Point: List of generated Point objects.
    """
    points_per_cluster = n_points // len(cluster_centers)
    data = []

    for i, center in enumerate(cluster_centers):
        x_center, y_center = center
        std_dev = cluster_stds[i]
        
        x_points = np.random.normal(loc=x_center, scale=std_dev, size=points_per_cluster)
        y_points = np.random.normal(loc=y_center, scale=std_dev, size=points_per_cluster)
        
        for x, y in zip(x_points, y_points):
            start_params = generate_params(start_type, start_hyperparams)
            end_params = generate_params(end_type, end_hyperparams)
            
            data.append({
                'x': x,
                'y': y,
                'start_type': start_type,
                'start_params': start_params,
                'end_type': end_type,
                'end_params': end_params
            })

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert DataFrame to list of Point objects
    points = df_to_pts(df)
    
    return points

def generate_params(dist_type, hyperparams):
    """
    Generate distribution parameters from hyperparameters.

    Parameters:
    dist_type (str): Distribution type ('norm', 'uniform', 'constant', 'calrcarbon').
    hyperparams (list of tuples): Hyperparameters for the distribution.
    calcurve_name (str, optional): Name of the calibration curve data required for 'calrcarbon' distribution.

    Returns:
    list: Generated distribution parameters.
    """
    params = []
    
    if dist_type == 'constant':
        params.append(hyperparams[0])
    elif dist_type == "norm":
        params.append(np.random.normal(loc=hyperparams[0], scale=hyperparams[1]))
        params.append(np.random.exponential(scale=hyperparams[2]))
    elif dist_type == "uniform":
        a = np.random.normal(loc=hyperparams[0], scale=hyperparams[1])
        params.append(a)
        params.append(a + np.random.exponential(scale=hyperparams[2]))
    elif dist_type == "calrcarbon":
        calcurve_name = hyperparams[0]
        calcurve = calibration_curves.get(calcurve_name)
        if calcurve is None:
            raise ValueError(f"Calibration curve {calcurve_name} not found")
        
        # Generate a tau value from the given distribution
        tau = np.random.normal(loc=hyperparams[1], scale=hyperparams[2])  # For example, tau could be generated from a normal distribution
        
        # Back-calibrate tau to get c14_mean and c14_err
        cal_rc = calrcarbon(calcurve)
        curve_mean, curve_error = cal_rc._calc_curve_params(tau)
        
        # Generate c14_mean as a normal distribution around the back-calibrated mean
        c14_mean = np.random.normal(loc=curve_mean, scale=curve_error)
        
        # Use the provided c14_err or a generated value
        c14_err = curve_error
        
        params = [calcurve_name, c14_mean, c14_err]
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")
        
    return params