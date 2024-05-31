# ChronoCluster

ChronoCluster is a Python package for analyzing spatial point patterns with temporality and chronological uncertainty, incorporating methods like Ripley's K function.

## Installation

To install the package, use:

```bash
pip install git+https://github.com/wccarleton/ChronoCluster.git
```

Alternatively, with Conda:

```bash
conda install -c wccarleton chronocluster
```

## Example: Generating and Analyzing Spatial Point Patterns

Here is a basic example of how to use the ChronoCluster package. For a full example, refer to the example.py script.

```python
import numpy as np
from chronocluster.data.simdata import generate_random_points
from chronocluster import clustering
from chronocluster.utils import clustering_heatmap, pdiff_heatmap, plot_mc_points, get_box
import matplotlib.pyplot as plt

# Define cluster distributions
cluster_centers = [(10, 10), (20, 20)]
cluster_stds = [1.0, 1.5]

# Define dating models and uncertainties
start_type = 'norm'
start_hyperparams = [1000, 50, 100]
end_type = 'constant'
end_hyperparams = [2000]

# Generate random points
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

# Get a bounding box
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

# Define time slices
start_time = 500
end_time = 1500
time_interval = 50
time_slices = np.arange(start_time, end_time, time_interval)

# Precompute inclusion probabilities based on age models and time slices
inclusion_probs = clustering.in_probs(points, time_slices)

# Run the Monte Carlo simulation to get an ensemble of probable 
# lists of points included in each time slice.
num_iterations = 100
simulations = clustering.mc_samples(points, 
                                    time_slices, 
                                    inclusion_probs, 
                                    num_iterations = num_iterations)

# Define distances for clustering stats
distances = np.linspace(np.finfo(float).eps, max_distance, num=40)

# Calculate K function over time
k_results, l_results, g_results = clustering.temporal_cluster(simulations, 
                                                            distances, 
                                                            time_slices, 
                                                            calc_K = True, 
                                                            calc_L = True,
                                                            calc_G = True)

# Plot the heatmap of mean K values (averaging over chronological uncertainty)
clustering_heatmap(k_results, distances, time_slices)
```

## Features

-Generate random points with spatial and temporal uncertainty.
-Visualize spatial point patterns.
-Perform clustering analysis using Ripley's K, L, and G functions while accounting for temporality (change over time) and chronological uncertainty.
-Compare observed data to CSR baselines.

## Contributing

Contributions are most welcome!

If you would like to contribute to ChronoCluster, please fork the repository and create a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.

## Contact

For any questions or comments, please contact W. Christopher Carleton at ccarleton@protonmail.com.