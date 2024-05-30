# Analysis
import numpy as np
from chronocluster.data.simdata import generate_random_points, generate_params
from chronocluster import clustering
from chronocluster.data.simdata import generate_random_points
from chronocluster.utils import clustering_heatmap, pdiff_heatmap, plot_mc_points

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

generate_params("uniform", hyperparams=(1000,50,100))

# Example usage
cluster_centers = [(10, 10), (20, 20)]
cluster_stds = [1.0, 1.5]
start_type = 'norm'
start_hyperparams = [1000, 50, 100]
end_type = 'constant'
end_hyperparams = [2000]

#end_type = 'norm'
#end_hyperparams = [1200, 50, 100]

points = generate_random_points(100, 
                                cluster_centers, 
                                cluster_stds, 
                                start_type, 
                                start_hyperparams, 
                                end_type, 
                                end_hyperparams)

# Print some of the generated points for debugging
for point in points[:5]:
    print(point)

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
end_time = 1500
time_interval = 50
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
                   t = 1)#np.where(time_slices == 1400)[0][0])

# Define distances for Ripley's K function
distances = np.linspace(1, 20, num=40)

# Calculate K function over time
k_results, l_results, g_results = clustering.temporal_cluster(simulations, 
                                                            distances, 
                                                            time_slices, 
                                                            calc_K = True, 
                                                            calc_L = True,
                                                            calc_G = True)

# Plot to see K(t,d) for a specific time slice
t_index = 9
plt.plot(distances, np.mean(l_results[:, t_index, :], axis=1))
plt.xlabel('Distance')
plt.ylabel('Normalized Ripley\'s K')
plt.title(f'Ripley\'s K Function for Time Slice {time_slices[t_index]}')
plt.show()

# Plot the heatmap of mean K values
clustering_heatmap(l_results, distances, time_slices)

# establish common support for evaluating densities by fixing max distance
max_distance = 20

pairwise_density, support = clustering.temporal_pairwise(simulations, 
                                                         time_slices, 
                                                         bw = 1, 
                                                         density = False, 
                                                         max_distance = max_distance)

clustering_heatmap(pairwise_density,
                    support,
                    time_slices,
                    result_type = 'K')