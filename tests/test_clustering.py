import pytest
import numpy as np
from chronocluster.clustering import temporal_cluster, temporal_pairwise, p_diff, mc_samples
from chronocluster.data.simdata import generate_random_points

def test_temporal_cluster():
    # Generate some sample data
    points = generate_random_points(100, [(5, 5), (15, 15)], [1.0, 1.5], 'norm', [10, 2, 5], 'uniform', [20, 3, 5])
    time_slices = np.linspace(0, 100, 10)
    simulations = mc_samples(points, time_slices, num_iterations = 100)
    
    distances = np.linspace(0, 10, 5)
    results = temporal_cluster(simulations, distances, time_slices, calc_K=True, calc_L=True, calc_G=True)
    
    # Assuming temporal_cluster returns three metrics
    k_results, l_results, g_results = results[:3]
    
    # Check the shapes of the results
    assert k_results.shape == (len(distances), len(time_slices), len(simulations))
    assert l_results.shape == (len(distances), len(time_slices), len(simulations))
    assert g_results.shape == (len(distances), len(time_slices), len(simulations))
    
    # Check the value ranges of the results
    assert np.all((k_results >= 0) | (np.isnan(k_results))), "Ripley's K values should be positive."
    assert np.all((l_results >= 0) | (np.isnan(l_results))), "Ripley's L values should be positive."
    assert np.all((g_results >= 0) | (np.isnan(g_results))), "Ripley's G values should be positive."

def test_temporal_pairwise():
    # Generate some sample data
    points = generate_random_points(100, [(5, 5), (15, 15)], [1.0, 1.5], 'norm', [10, 2, 5], 'uniform', [20, 3, 5])
    time_slices = np.linspace(0, 100, 10)
    simulations = mc_samples(points, time_slices, num_iterations = 100)
    
    bw = 1.0
    max_distance = 10.0
    
    results = temporal_pairwise(simulations, time_slices, bw, density=False, max_distance=max_distance)
    
    # temporal_pairwise returns two objects
    pairwise_density, support = results[:2]
    
    # Check the shape of the pairwise density result
    assert pairwise_density.shape == (len(time_slices), len(time_slices), len(simulations))
    assert len(support) == len(time_slices)
    
    # Check the value ranges of the pairwise density result
    assert np.all((pairwise_density >= 0) | (np.isnan(pairwise_density))), "Pairwise density values should be positive."

def test_p_diff():
    # Generate some sample data
    distances = 5
    time_slices = 10
    iterations = 100
    pairwise_density = np.random.rand(distances, time_slices, iterations)
    csr_pairwise_density = np.random.rand(distances, time_slices, iterations)
    
    p_values = p_diff(pairwise_density, csr_pairwise_density, condition='greater')
    
    # Check the shape of the p-values result
    assert p_values.shape == (distances, time_slices)
    
    # Check the value ranges of the p-values result
    assert np.all((0 <= p_values) & (p_values <= 1)), "p-values should be in the range [0, 1]."

if __name__ == '__main__':
    pytest.main()
