import numpy as np
import chronocluster
from chronocluster import clustering
from chronocluster.data.simdata import generate_random_points
from chronocluster.utils import clustering_heatmap, plot_l_diff, plot_mc_points

import numpy as np
import matplotlib.pyplot as plt
import importlib


importlib.reload(chronocluster.clustering)
importlib.reload(chronocluster.utils)
importlib.reload(chronocluster.data.simdata)

# Generate random points

n_points = 1000
cluster_centers = [(50, 50), (150, 150), (50, 150), (150, 50), (80, 80), (180, 180), (80, 180), (180, 80), (65, 65), (165, 165), (65, 165), (165, 65)]
cluster_std = 3

points = generate_random_points(n_points, cluster_centers, cluster_std)

# Visualize the generated points
x_coords = [p[0] for p in points]
y_coords = [p[1] for p in points]
plt.scatter(x_coords, y_coords, alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Generated Random Points')
plt.show()

# Define the time slices
start_time = 500
end_time = 2100
time_interval = 100
time_slices = np.arange(start_time, end_time, time_interval)

# Precompute inclusion probabilities
inclusion_probs = clustering.in_probs(points, time_slices, end_time)

# Run the Monte Carlo simulation
num_iterations = 10
simulations = clustering.mc_samples(points, time_slices, inclusion_probs, num_iterations=num_iterations)

# Define distances for Ripley's K function
distances = np.linspace(1, 200, num=40)

# Calculate K function over time
k_results, l_results, g_results = clustering.temporal_cluster(simulations, 
                                                            distances, 
                                                            time_slices, 
                                                            num_iterations = 10, 
                                                            calc_K = True, 
                                                            calc_L = True,
                                                            calc_G = True)

# Plot the heatmap of mean K values
clustering_heatmap(k_results, distances, time_slices)

# Additional plot to see K(t,d) for a specific time slice
t_index = 9  # For example, the 6th time slice
plt.plot(distances, np.mean(l_results[:, t_index, :], axis=1))
plt.xlabel('Distance')
plt.ylabel('Normalized Ripley\'s K')
plt.title(f'Ripley\'s K Function for Time Slice {time_slices[t_index]}')
plt.show()

# establish common support for evaluating densities by fixing max distance
max_distance = 200

pairwise_density, support = clustering.temporal_pairwise(simulations, time_slices, bw=1, density=False, max_distance = max_distance)

t_index = 9
plt.plot(support, np.mean(pairwise_density[:, t_index, :], axis=1))
plt.xlabel('Distance')
plt.ylabel('Density of Pairwise Distances')
plt.title(f'Pairewise Distances for Time Slice {time_slices[t_index]}')
plt.show()

clustering_heatmap(pairwise_density, support, time_slices)

plot_mc_points(simulations, iter = 5, t = 10)

# CSR baseline
# Generate CSR sample
x_min, x_max = 1, 200
y_min, y_max = 1, 200
csr_points = clustering.csr_sample(points, x_min, x_max, y_min, y_max)

# Visualize the CSR sample
csr_x_coords = [p[0] for p in csr_points]
csr_y_coords = [p[1] for p in csr_points]
plt.scatter(csr_x_coords, csr_y_coords, alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('CSR Sampled Points')
plt.show()

inclusion_probs = clustering.in_probs(csr_points, time_slices, end_time)

num_iterations = 10
csr_simulations = clustering.mc_samples(csr_points, time_slices, inclusion_probs, num_iterations=num_iterations)

csr_k_results, csr_l_results, csr_g_results = clustering.temporal_cluster(csr_simulations, 
                                                            distances, 
                                                            time_slices, 
                                                            num_iterations = 10, 
                                                            calc_K = True, 
                                                            calc_L = True,
                                                            calc_G = True)

# Additional plot to see K(t,d) for a specific time slice
t_index = 9  # For example, the 6th time slice
plt.plot(distances, np.mean(csr_k_results[:, t_index, :], axis=1))
plt.xlabel('Distance')
plt.ylabel('Normalized Ripley\'s K')
plt.title(f'Ripley\'s K Function for Time Slice {time_slices[t_index]}')
plt.show()

csr_pairwise_density, csr_support = clustering.temporal_pairwise(csr_simulations, time_slices, bw=1, density=False, max_distance=max_distance)

t_index = 9
plt.plot(csr_support, np.mean(csr_pairwise_density[:, t_index, :], axis=1))
plt.xlabel('Distance')
plt.ylabel('Density of Pairwise Distances')
plt.title(f'Pairewise Distances for Time Slice {time_slices[t_index]}')
plt.show()

### compare 
t_index = 9
plt.plot(csr_support, np.mean(pairwise_density[:, t_index, :], axis=1) - np.mean(csr_pairwise_density[:, t_index, :], axis=1))
plt.xlabel('Distance')
plt.ylabel('Density of Pairwise Distances')
plt.title(f'Pairewise Distances for Time Slice {time_slices[t_index]}')
plt.show()

## difference
density_diff = pairwise_density - csr_pairwise_density
# Assume density_diff is already computed and has shape (distances, time_slices, iterations)
# Example support and time_slices for plotting
# support = np.linspace(0, 10, 100)  # Example distances
time_slice_index = 15  # Change as needed

# Extract the data for the given time slice
time_slice_data = density_diff[:, time_slice_index, :]

# Calculate the mean, 5th percentile, and 95th percentile
mean_density_diff = np.mean(time_slice_data, axis=1)
quantile_5 = np.percentile(time_slice_data, 5, axis=1)
quantile_95 = np.percentile(time_slice_data, 95, axis=1)

# Plot the mean and the quantile range
plt.figure(figsize=(10, 6))
plt.plot(support, mean_density_diff, label='Mean Density Difference')
plt.fill_between(support, quantile_5, quantile_95, color='gray', alpha=0.5, label='5th-95th Percentile Range')
plt.xlabel('Pairwise Distances')
plt.ylabel('Density Difference')
plt.title(f'Density Difference at Time Slice {time_slice_index}')
plt.legend()
plt.show()

