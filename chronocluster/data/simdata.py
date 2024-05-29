import numpy as np
import matplotlib.pyplot as plt
from chronocluster.clustering import Point
from scipy.stats import norm

def generate_random_points(n_points, 
                           cluster_centers, 
                           cluster_std, 
                           age_mean = 1000, 
                           age_sd = 200,
                           age_error = 50,
                           verbose=False):
    """
    Generate random Point objects with (x, y, start_distribution, end_distribution).
    
    Parameters:
    n_points (int): Number of points to generate.
    cluster_centers (list of tuples): Centers of the clusters [(x1, y1), (x2, y2), ...].
    cluster_std (float): Standard deviation of the clusters.
    age_mean (float): Mean of the ages.
    age_sd (float): Standard deviation of the ages.
    age_error (float): Standard deviation of the age error.
    verbose (bool): If True, prints temporal consistency check messages.
    
    Returns:
    list: A list of Point objects.
    """
    points = []
    n_clusters = len(cluster_centers)
    points_per_cluster = n_points // n_clusters
    
    for center in cluster_centers:
        x_center, y_center = center
        x_points = np.random.normal(loc=x_center, scale=cluster_std, size=points_per_cluster)
        y_points = np.random.normal(loc=y_center, scale=cluster_std, size=points_per_cluster)
        mean_ages = np.random.normal(loc=age_mean, scale=age_sd, size=points_per_cluster)
        sd_ages = np.full(points_per_cluster, age_error)  # Assuming a fixed standard deviation for ages

        for x, y, mean, sd in zip(x_points, y_points, mean_ages, sd_ages):
            start_distribution = norm(loc=mean, scale=sd)
            end_distribution = norm(loc=mean + 100, scale=sd)  # Example: end date is 100 units after start date
            points.append(Point(x, y, start_distribution, end_distribution, verbose=verbose))
    
    return points
