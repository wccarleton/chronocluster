from chronocluster.distributions import calrcarbon
from chronocluster.calcurves import calibration_curves
from chronocluster.data.simdata import generate_random_points

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