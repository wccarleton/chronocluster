import numpy as np
from chronocluster import clustering
from chronocluster.data.simdata import generate_random_points
from chronocluster.utils import clustering_heatmap, pdiff_heatmap
from scipy.stats import norm, uniform


# Example usage with SciPy distributions
points = [
    clustering.Point(10, 20, norm(loc=100, scale=10), norm(loc=90, scale=10), verbose=True),  # Edge case: end before start
    clustering.Point(15, 25, norm(loc=100, scale=10), norm(loc=200, scale=10), verbose=True),  # Normal case
    clustering.Point(20, 30, uniform(loc=100, scale=50), uniform(loc=80, scale=50), verbose=True)  # Edge case: end before start
]

time_slices = np.arange(50, 300, 10)

inclusion_probs = clustering.in_probs(points, time_slices)

# Display the inclusion probabilities for debugging
print(inclusion_probs)