# analytical packages
import numpy as np
from chronocluster import clustering
from chronocluster.data.simdata import generate_random_points
from chronocluster.utils import clustering_heatmap, pdiff_heatmap, plot_mc_points

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Generate random points
n_points = 1000
cluster_centers = [(50, 50), 
                   (150, 150), 
                   (50, 150), 
                   (150, 50), 
                   (80, 80), 
                   (180, 180), 
                   (80, 180), 
                   (180, 80), 
                   (65, 65), 
                   (165, 165), 
                   (65, 165), 
                   (165, 65)]
cluster_std = 3

points = generate_random_points(n_points, 
                                cluster_centers, 
                                cluster_std, 
                                age_mean = 1000, 
                                age_sd = 50,
                                age_error = 10,
                                verbose=False)

# Visualize the generated points
x_coords = [point.x for point in points]
y_coords = [point.y for point in points]
plt.scatter(x_coords, y_coords, alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Generated Random Points')
plt.show()

# Define the time slices
start_time = 100
end_time = 2100
time_interval = 100
time_slices = np.arange(start_time, end_time, time_interval)

# Precompute inclusion probabilities
inclusion_probs = clustering.in_probs(points, time_slices)

# Run the Monte Carlo simulation
num_iterations = 100
simulations = clustering.mc_samples(points, 
                                    time_slices, 
                                    inclusion_probs, 
                                    num_iterations = num_iterations)

plot_mc_points(simulations, 
                   iter = 0, 
                   t = 9)

# Define distances for Ripley's K function
distances = np.linspace(1, 200, num=40)

# Calculate K function over time
k_results, l_results, g_results = clustering.temporal_cluster(simulations, 
                                                            distances, 
                                                            time_slices, 
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

# Now, pairwise distance distributions.

# establish common support for evaluating densities by fixing max distance
max_distance = 200

pairwise_density, support = clustering.temporal_pairwise(simulations, 
                                                         time_slices, 
                                                         bw = 1, 
                                                         density = False, 
                                                         max_distance = max_distance)

clustering_heatmap(pairwise_density,
                    support,
                    time_slices,
                    result_type = 'K')

# CSR baseline---the following is for comparing the observed point data to a comparable set 
# (same number of points, same chronological/temporal traits) but representing 
# Compelte Spatial Randomness (CSR)

# Generate CSR sample
x_min, x_max = 1, 200
y_min, y_max = 1, 200
csr_points = clustering.csr_sample(points, x_min, x_max, y_min, y_max)

# Visualize the CSR sample
csr_x_coords = [point.x for point in csr_points]
csr_y_coords = [point.y for point in csr_points]
plt.scatter(csr_x_coords, csr_y_coords, alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('CSR Sampled Points')
plt.show()

csr_inclusion_probs = clustering.in_probs(csr_points, 
                                          time_slices, 
                                          end_time)

csr_simulations = clustering.mc_samples(csr_points, 
                                        time_slices, 
                                        inclusion_probs, 
                                        num_iterations = num_iterations)

csr_k_results, csr_l_results, csr_g_results = clustering.temporal_cluster(csr_simulations, 
                                                            distances, 
                                                            time_slices, 
                                                            calc_K = True, 
                                                            calc_L = True,
                                                            calc_G = True)

csr_pairwise_density, csr_support = clustering.temporal_pairwise(csr_simulations, 
                                                                 time_slices, 
                                                                 bw = 1, 
                                                                 density = False, 
                                                                 max_distance = max_distance)

# Ensure the shapes are the same
assert pairwise_density.shape == csr_pairwise_density.shape

# Calculate the p-values for density differences between the observed points and 
# the simulated CSR baseline per distance and temporal slice
p_diff_array = clustering.p_diff(pairwise_density, csr_pairwise_density)

# Plot the heatmap of probabilities
pdiff_heatmap(p_diff_array, time_slices, support)

