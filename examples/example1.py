import numpy as np
from chronocluster import clustering
from chronocluster.data.simdata import generate_random_points
from chronocluster.utils import clustering_heatmap
import numpy as np
import matplotlib.pyplot as plt

# Generate random points
n_points = 100
cluster_centers = [(50, 50), (150, 150)]
cluster_std = 10

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
start_time = 100
end_time = 2100
time_interval = 100
time_slices = np.arange(start_time, end_time, time_interval)

# Precompute inclusion probabilities
inclusion_probs = clustering.in_probs(points, time_slices, end_time)

# Run the Monte Carlo simulation
num_iterations = 10
simulations = clustering.mc_samples(points, time_slices, inclusion_probs, num_iterations=num_iterations)

# Define distances for Ripley's K function
distances = np.linspace(1, 200, num=10)

# Calculate K function over time
k_results = clustering.temporal_k(simulations, distances, num_iterations, time_slices)

# Plot the heatmap of mean K values
clustering_heatmap(k_results, time_slices, distances)

# Additional plot to see K(t,d) for a specific time slice
t_index = 5  # For example, the 6th time slice
plt.plot(distances, np.mean(k_results[:, t_index, :], axis=1))
plt.xlabel('Distance')
plt.ylabel('Normalized Ripley\'s K')
plt.title(f'Ripley\'s K Function for Time Slice {time_slices[t_index]}')
plt.show()

# Estimate the pair correlation function g(d)
g_results = clustering.temporal_pcor(k_results, distances)

# Plot the heatmap of mean g values
clustering_heatmap(g_results, time_slices, distances, result_type='g')

# Plot g(d) for an arbitrary time slice
t_index = 5  # For example, the 6th time slice
mean_g_values = np.mean(g_results[:, t_index, :], axis=1)
std_g_values = np.std(g_results[:, t_index, :], axis=1)

plt.figure(figsize=(8, 6))
plt.plot(distances, mean_g_values, label='g(d)')
plt.fill_between(distances, mean_g_values - std_g_values, mean_g_values + std_g_values, alpha=0.3)
plt.xlabel('Distance')
plt.ylabel('Pair Correlation Function g(d)')
plt.title(f"Pair Correlation Function for Time Slice {time_slices[t_index]}")
plt.legend()
plt.show()

# Plot K(d) and its derivative g(d) for an arbitrary time slice
t_index = 5  # For example, the 6th time slice
mean_k_values = np.mean(k_results[:, t_index, :], axis=1)
std_k_values = np.std(k_results[:, t_index, :], axis=1)

# Calculate the derivative g(d)
g_values = np.gradient(mean_k_values, distances)

plt.figure(figsize=(8, 6))
plt.plot(distances, mean_k_values, label='Mean K(d)')
plt.fill_between(distances, mean_k_values - std_k_values, mean_k_values + std_k_values, alpha=0.3)
plt.xlabel('Distance')
plt.ylabel('Normalized Ripley\'s K')
plt.title(f"Ripley's K Function for Time Slice {time_slices[t_index]}")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(distances, g_values, label='g(d)')
plt.xlabel('Distance')
plt.ylabel('Pair Correlation Function g(d)')
plt.title(f"Pair Correlation Function for Time Slice {time_slices[t_index]}")
plt.legend()
plt.show()