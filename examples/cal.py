from chronocluster.distributions import calrcarbon
from chronocluster.calcurves import calibration_curves
from chronocluster.data.simdata import generate_random_points
from chronocluster import clustering
from chronocluster.utils import clustering_heatmap, pdiff_heatmap, plot_mc_points, get_box

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# plotting
from chronocluster.utils import calrc_plot

points = generate_random_points(
    n_points=100,
    cluster_centers=[(0, 0), (10, 10)],
    cluster_stds=[1.0, 2.0],
    start_type='calrcarbon',
    start_hyperparams=['intcal20', -5000, 200],  # Using the calibration curve name
    end_type='norm',
    end_hyperparams=[0, 1, 1]
)

calrc_plot(points[1].start_distribution, plot_type='pdf', bins=100)

# Get a bounding box for use later and to extract sensible distance limits
x_min, y_min, x_max, y_max = get_box(points)

max_distance = np.ceil(np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2))

# Define the time slices
start_time = -6000
end_time = -5000
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