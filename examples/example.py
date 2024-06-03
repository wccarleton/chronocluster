# Analysis
import numpy as np
from chronocluster.data.simdata import generate_random_points
from chronocluster import clustering
from chronocluster.data.simdata import generate_random_points
from chronocluster.utils import clustering_heatmap, pdiff_heatmap, plot_mc_points, get_box, ddelta

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Example usage
# Generate random point locations in clusters
# define cluster distributions
cluster_centers = [(10, 10), (20, 20)]
cluster_stds = [1.0, 1.5]

# create random start and end ages with uncertainty for each point
# define dating models and uncertainties
start_type = 'norm'
start_hyperparams = [1000, 50, 100]
end_type = 'constant'
end_hyperparams = [2000]

# finally, generate 100 random points using the above models
points = generate_random_points(100, 
                                cluster_centers, 
                                cluster_stds, 
                                start_type, 
                                start_hyperparams, 
                                end_type, 
                                end_hyperparams)

# Print some of the generated points
for point in points[:5]:
    print(point)

# Get a bounding box for use later and to extract sensible distance limits
x_min, y_min, x_max, y_max = get_box(points)

max_distance = np.ceil(np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2))

# Visualize the generated points
x_coords = [point.x for point in points]
y_coords = [point.y for point in points]
plt.scatter(x_coords, y_coords, alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Generated Random Points')
plt.show()

# Define the time slices
start_time = 500
end_time = 1500
time_interval = 50
time_slices = np.arange(start_time, end_time, time_interval)

# Precompute inclusion probabilities based on age models and time slices
inclusion_probs = clustering.in_probs(points, time_slices)

# see what these look like for a randomly selected point
random_point_idx = int(np.random.uniform(low = 0, high = 99))
rnd_inclusion_probs = inclusion_probs[random_point_idx, :]

plt.vlines(time_slices, 
           ymin = 0, 
           ymax = rnd_inclusion_probs, 
           color = 'blue', 
           label = 'Inclusion Probabilities')
plt.xlabel('Time Slice')
plt.ylabel('Inclusion Probability')
plt.title('Inclusion Probabilities per Time Slice')
plt.legend()
plt.show()

# Run the Monte Carlo simulation to get an ensemble of probable 
# lists of points included in each time slice.
num_iterations = 100
simulations = clustering.mc_samples(points, 
                                    time_slices, 
                                    inclusion_probs, 
                                    num_iterations = num_iterations)

# Plot an MC iteration for a given time_slice index---this is just to 
# see what the point distribution at the given time, for the given
# MC iteration looks like

rnd_t = int(np.random.uniform(low = 0, high = len(time_slices)))
rnd_iter = int(np.random.uniform(low = 0, high = num_iterations))
plot_mc_points(simulations, 
                   iter = rnd_iter, 
                   t = rnd_t)

# Define distances for Ripley's K function
distances = np.linspace(np.finfo(float).eps, max_distance, num=40)

# Calculate K function over time
k_results, l_results, g_results = clustering.temporal_cluster(simulations, 
                                                            distances, 
                                                            time_slices, 
                                                            calc_K = True, 
                                                            calc_L = True,
                                                            calc_G = True)

# Plot to see results for a random time slice
rnd_t = int(np.random.uniform(low = 0, high = len(time_slices)))

# Create a 3-panel plot
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot K results
axs[0].plot(distances, np.mean(k_results[:, rnd_t, :], axis=1))
axs[0].set_xlabel('Distance')
axs[0].set_ylabel('Normalized Ripley\'s K')
axs[0].set_title(f'Ripley\'s K Function for Time Slice {time_slices[rnd_t]}')

# Plot L results
axs[1].plot(distances, np.mean(l_results[:, rnd_t, :], axis=1))
axs[1].set_xlabel('Distance')
axs[1].set_ylabel('Normalized Ripley\'s L')
axs[1].set_title(f'Ripley\'s L Function for Time Slice {time_slices[rnd_t]}')

# Plot G results
axs[2].plot(distances, np.mean(g_results[:, rnd_t, :], axis=1))
axs[2].set_xlabel('Distance')
axs[2].set_ylabel('Normalized Ripley\'s G')
axs[2].set_title(f'Ripley\'s G Function for Time Slice {time_slices[rnd_t]}')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()

# Plot the heatmap of mean K values (averaging over chronological uncertainty)
clustering_heatmap(k_results, distances, time_slices)

# Produce pairwise distances to explore clustering structure
pairwise_density, support = clustering.temporal_pairwise(simulations, 
                                                         time_slices, 
                                                         bw = 1, 
                                                         density = False, 
                                                         max_distance = max_distance)

clustering_heatmap(pairwise_density,
                    support,
                    time_slices,
                    result_type = 'Pairwise Distances')

# Comapre the empirical pairwise distances to complete spatial randomness (CSR)
# taking into account chronological uncertainty---the difference between the
# distributions reflects the complete distribution of differences between
# a CSR simlulated dataset and each of the MC iterations reflecting different
# probable time series of point sets.

# CSR baseline---the following is for comparing the observed point data to a 
# comparable set (same number of points, same chronological/temporal traits) but 
# representing Compelte Spatial Randomness (CSR)

# Generate CSR sample

csr_points = clustering.csr_sample(points, x_min, x_max, y_min, y_max)

# Visualize the CSR sample
csr_x_coords = [point.x for point in csr_points]
csr_y_coords = [point.y for point in csr_points]
plt.scatter(csr_x_coords, csr_y_coords, alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('CSR Sampled Points')
plt.show()

# Get CSR inclusion probabilities (remember these points have the same
# temporal traits as the first simulated data)
csr_inclusion_probs = clustering.in_probs(csr_points, 
                                          time_slices, 
                                          end_time)

# Get MC iterations for incorporating chronological uncertainty
csr_simulations = clustering.mc_samples(csr_points, 
                                        time_slices, 
                                        inclusion_probs, 
                                        num_iterations = num_iterations)

# Calculate the CSR version of the same clustering stats
csr_k_results, csr_l_results, csr_g_results = clustering.temporal_cluster(csr_simulations, 
                                                            distances, 
                                                            time_slices, 
                                                            calc_K = True, 
                                                            calc_L = True,
                                                            calc_G = True)

# Calulate the pairwise distances for the CSR sample
csr_pairwise_density, csr_support = clustering.temporal_pairwise(csr_simulations, 
                                                                 time_slices, 
                                                                 bw = 1, 
                                                                 density = False, 
                                                                 max_distance = max_distance)

# Ensure the array shapes are the same
assert pairwise_density.shape == csr_pairwise_density.shape

# Calculate the p-values for density differences between the observed points and 
# the simulated CSR baseline per distance and temporal slice
p_diff_array = clustering.p_diff(pairwise_density, csr_pairwise_density)

# Plot the heatmap of probabilities
pdiff_heatmap(p_diff_array, time_slices, support)

# Now consider an analysis using focal points (do points cluster around given
# focal points?):

# Parameters for ddelta
start_d = 500
end_d = 1500

# Generate focal points
focal_points = []
for x, y in cluster_centers:
    focal_points.append(clustering.Point(x = x, 
                              y = y, 
                              start_distribution = ddelta(start_d), 
                              end_distribution = ddelta(end_d)))

# Create focal point mc_simulations
focal_simulations = clustering.mc_samples(focal_points, time_slices, num_iterations=num_iterations)

focal_simulations[0][0][1].shape

# Calculate K-function for focal points
focal_k_results, focal_l_results, focal_g_results = clustering.temporal_cluster(
    simulations, distances, time_slices, calc_K=True, calc_L=True, calc_G=True, focal_points=focal_simulations)

# Calculate pairwise distances for focal points
focal_pairwise_density, focal_support = clustering.temporal_pairwise(
    simulations, time_slices, bw=1.0, density=False, max_distance=max_distance, focal_points=focal_simulations)

# Plot to see results for a random time slice for focal points
rnd_t = int(np.random.uniform(low = 0, high = len(time_slices)))

# Create a 3-panel plot for focal points
fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# Plot K results
axs[0].plot(distances, np.mean(focal_k_results[:, rnd_t, :], axis=1))
axs[0].set_xlabel('Distance')
axs[0].set_ylabel('Normalized Ripley\'s K')
axs[0].set_title(f'Ripley\'s K Function for Time Slice {time_slices[rnd_t]} (Focal Points)')

# Plot L results
axs[1].plot(distances, np.mean(focal_l_results[:, rnd_t, :], axis=1))
axs[1].set_xlabel('Distance')
axs[1].set_ylabel('Normalized Ripley\'s L')
axs[1].set_title(f'Ripley\'s L Function for Time Slice {time_slices[rnd_t]} (Focal Points)')

# Plot G results
axs[2].plot(distances, np.mean(focal_g_results[:, rnd_t, :], axis=1))
axs[2].set_xlabel('Distance')
axs[2].set_ylabel('Normalized Ripley\'s G')
axs[2].set_title(f'Ripley\'s G Function for Time Slice {time_slices[rnd_t]} (Focal Points)')

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()

# Plot the heatmap of mean K values (averaging over chronological uncertainty) for focal points
clustering_heatmap(focal_k_results, distances, time_slices)

# Produce pairwise distances to explore clustering structure for focal points
clustering_heatmap(focal_pairwise_density, focal_support, time_slices, result_type='Pairwise Distances (Focal Points)')