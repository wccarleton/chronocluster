import numpy as np
import matplotlib.pyplot as plt

def generate_random_points(n_points, cluster_centers, cluster_std, age_mean=1000, age_sd=200):
    """
    Generate random points (x, y, mean_age, sd_age).
    
    Parameters:
    n_points (int): Number of points to generate.
    cluster_centers (list of tuples): Centers of the clusters [(x1, y1), (x2, y2), ...].
    cluster_std (float): Standard deviation of the clusters.
    age_mean (float): Mean of the ages.
    age_sd (float): Standard deviation of the ages.
    
    Returns:
    list: A list of points [x, y, mean_age, sd_age].
    """
    points = []
    n_clusters = len(cluster_centers)
    points_per_cluster = n_points // n_clusters
    
    for center in cluster_centers:
        x_center, y_center = center
        x_points = np.random.normal(loc=x_center, scale=cluster_std, size=points_per_cluster)
        y_points = np.random.normal(loc=y_center, scale=cluster_std, size=points_per_cluster)
        mean_ages = np.random.normal(loc=age_mean, scale=age_sd, size=points_per_cluster)
        sd_ages = np.full(points_per_cluster, 50)  # Assuming a fixed standard deviation for ages

        for x, y, mean, sd in zip(x_points, y_points, mean_ages, sd_ages):
            points.append([x, y, mean, sd])
    
    return points
