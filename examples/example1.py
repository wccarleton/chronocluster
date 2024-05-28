import numpy as np
from chronocluster import clustering
from chronocluster.data.simdata import generate_random_points
from chronocluster.utils import clustering_heatmap, pdiff_heatmap

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

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
num_iterations = 100
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

csr_inclusion_probs = clustering.in_probs(csr_points, time_slices, end_time)
csr_simulations = clustering.mc_samples(csr_points, time_slices, inclusion_probs, num_iterations=num_iterations)

csr_k_results, csr_l_results, csr_g_results = clustering.temporal_cluster(csr_simulations, 
                                                            distances, 
                                                            time_slices, 
                                                            num_iterations = 10, 
                                                            calc_K = True, 
                                                            calc_L = True,
                                                            calc_G = True)

csr_pairwise_density, csr_support = clustering.temporal_pairwise(csr_simulations, time_slices, bw=1, density=False, max_distance=max_distance)

# Ensure the shapes are the same
assert pairwise_density.shape == csr_pairwise_density.shape

p_diff_array = clustering.p_diff(pairwise_density, csr_pairwise_density)

# Plot the heatmap of probabilities
plt.figure(figsize=(12, 6))
sns.heatmap(p_diff_array, xticklabels=time_slices, yticklabels=support, cmap='viridis', cbar_kws={'label': 'P(diff > 0)'})
plt.xlabel('Time Slices')
plt.ylabel('Pairwise Distances')
plt.title('Heatmap of P(diff > 0) Over Time and Distance')

# Adjust y-axis ticks and invert the axis
num_ticks = 10  # Desired number of y-ticks
tick_indices = np.linspace(0, len(support) - 1, num_ticks, dtype=int)
tick_labels = np.round(support[tick_indices], 2)
plt.yticks(tick_indices, tick_labels)
plt.gca().invert_yaxis()

plt.show()

pdiff_heatmap(p_diff_array, time_slices, support)

