#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Christopher Carleton & Shuang Song
# @Contact   : carleton@gea.mpg.de
# GitHub   : https://github.com/wccarleton/chronocluster

"""
Test tutorial

This is a test for the tutorial notebook.
Three sessions are included:
1. Space-time plot of points and their distribution, density.
2. Plotting the PDF and CDF of the calibration curve
3. Plotting multiple points
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from chronocluster import clustering
from chronocluster.clustering import mc_samples
from chronocluster.density import custom_kde, gmm_peak_finder, kde_peaks, kde_time
from chronocluster.utils import (
    calrc_plot,
    chrono_plot,
    clustering_heatmap,
    get_box,
    plot_pdd,
)

IMG_DIR = "baseline_images"
TOL = 10  # 增加容差值，默认是2


class TestPoints:
    """
    Test points
    """

    @pytest.mark.mpl_image_compare(
        baseline_dir=IMG_DIR, filename="tutorial_plot.png", tolerance=TOL
    )
    def test_tutorial_points(self, point, style_params):
        """
        Test tutorial points
        """
        # Custom styling parameters
        ax, _ = chrono_plot(point, style_params=style_params, time_slice=1100)
        ax.set_box_aspect(None, zoom=0.85)
        return ax.figure  # return figure object

    @pytest.mark.mpl_image_compare(
        baseline_dir=IMG_DIR, filename="tutorial_calrc_pdf_plot.png"
    )
    def test_tutorial_calrc_pdf_plot(self, curve):
        """
        Test tutorial calrc pdf plot
        """
        # Plot PDF and histogram
        calrc_plot(curve, plot_type="pdf", bins=100)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(
        baseline_dir=IMG_DIR, filename="tutorial_calrc_cdf_plot.png"
    )
    def test_tutorial_calrc_cdf_plot(self, curve):
        """
        Test tutorial calrc cdf plot
        """
        # Plot CDF and ECDF
        calrc_plot(curve, plot_type="cdf", bins=100)
        return plt.gcf()

    @pytest.mark.mpl_image_compare(
        baseline_dir=IMG_DIR, filename="tutorial_multiple_points.png"
    )
    def test_tutorial_multiple_points(self, points, style_params):
        """
        Test tutorial multiple points
        """
        # Custom styling parameters
        ax, _ = chrono_plot(points, style_params=style_params, time_slice=1100)
        ax.set_box_aspect(None, zoom=0.85)
        return ax.figure  # return figure object


class TestPairwise:
    """
    Test pairwise density calculation
    """

    @pytest.fixture(name="max_distance")
    def max_distance_fixture(self, points):
        """
        Fixture for max distance of points.
        """
        # Get a bounding box for use later and to extract sensible distance limits
        x_min, y_min, x_max, y_max = get_box(points)
        return np.ceil(np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2))

    @pytest.fixture(name="pairwise_result")
    def pairwise_result_fixture(self, points, time_slices, max_distance):
        """
        Fixture for pairwise density calculation
        """
        simulations = mc_samples(points, time_slices, num_iterations=100)
        pairwise_density, support = clustering.temporal_pairwise(
            simulations,
            time_slices,
            bw=0.5,
            use_kde=True,
            max_distance=max_distance,
        )
        return pairwise_density, support

    @pytest.fixture(name="pairwise_density")
    def pairwise_density_fixture(self, pairwise_result):
        """
        Fixture for pairwise density
        """
        return pairwise_result[0]

    @pytest.fixture(name="support")
    def support_fixture(self, pairwise_result):
        """
        Fixture for support
        """
        return pairwise_result[1]

    @pytest.mark.mpl_image_compare(
        baseline_dir=IMG_DIR,
        filename="tutorial_pairwise_density.png",
        tolerance=TOL,
    )
    def test_pairwise_density(self, pairwise_density, support, time_slices):
        """
        Test pairwise density calculation
        """
        # Visualize clustering with heatmap
        clustering_heatmap(
            pairwise_density,
            support,
            time_slices,
            result_type="Pairwise Distances",
        )
        return plt.gcf()

    # corresponding to time 1100
    @pytest.mark.mpl_image_compare(
        baseline_dir=IMG_DIR,
        filename="tutorial_pairwise_pdd.png",
        tolerance=TOL,
    )
    @pytest.mark.parametrize("time_profile", [1100])
    def test_pairwise_pdd(self, pairwise_density, support, time_slices, time_profile):
        """
        Test pairwise pdd calculation
        """
        time_slice_idx = np.where(time_slices == time_profile)[0][0]

        # List of density arrays
        density_arrays = [pairwise_density]

        # Generate the plot and get the figure and axis objects
        fig, ax = plot_pdd(
            time_slices=time_slices,
            time_slice_idx=time_slice_idx,
            support=support,
            density_arrays=density_arrays,
            quantiles=[0.025, 0.975],
            density_names=["Empirical"],
            colors=["blue"],
        )
        return fig

    @pytest.fixture(name="csr_points")
    def csr_points_fixture(self, points):
        """
        Fixture for csr points
        """
        # Get a bounding box for use later and to extract sensible distance limits
        x_min, y_min, x_max, y_max = get_box(points)
        # Generate one sample of CSR from the points list for plotting
        csr_points, _ = clustering.csr_sample(
            points, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        )
        return csr_points

    @pytest.mark.mpl_image_compare(
        baseline_dir=IMG_DIR,
        filename="tutorial_random_points.png",
        tolerance=TOL,
    )
    def test_random_points(self, csr_points, style_params):
        """
        Test random points
        """
        ax, _ = chrono_plot(csr_points, style_params=style_params, time_slice=1100)
        ax.set_box_aspect(None, zoom=0.85)
        return ax.figure

    @pytest.fixture(name="csr_simulations")
    def csr_simulations_fixture(self, points, time_slices, max_distance):
        """
        Fixture for CSR simulations
        """
        x_min, y_min, x_max, y_max = get_box(points)
        csr_simulations = clustering.mc_samples(
            points=points,
            time_slices=time_slices,
            num_iterations=100,
            null_model=clustering.csr_sample,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
        )

        # Calulate the pairwise distances for the CSR sample
        return clustering.temporal_pairwise(
            csr_simulations,
            time_slices,
            bw=0.5,
            use_kde=True,
            max_distance=max_distance,
        )

    @pytest.mark.mpl_image_compare(
        baseline_dir=IMG_DIR,
        filename="tutorial_random_pairwise.png",
        tolerance=TOL,
    )
    def test_random_pairwise(self, csr_simulations, time_slices):
        """
        Test random pairwise
        """
        csr_pairwise_density, csr_support = csr_simulations

        # Visualize clustering with heatmap
        clustering_heatmap(
            csr_pairwise_density,
            csr_support,
            time_slices,
            result_type="Pairwise Distances",
        )
        return plt.gcf()

    @pytest.mark.mpl_image_compare(
        baseline_dir=IMG_DIR,
        filename="tutorial_random_pairwise_pdd.png",
        tolerance=TOL,
    )
    def test_random_pairwise_pdd(
        self, pairwise_density, support, time_slices, csr_simulations
    ):
        csr_pairwise_density, csr_support = csr_simulations
        time_slice_idx = np.where(time_slices == 1100)[0][
            0
        ]  # corresponding to time 1100
        time_slice = time_slices[time_slice_idx]

        # Define quantiles for the error envelope
        quantiles = [0.025, 0.975]  # 95% confidence interval

        # Calculate mean and quantiles for empirical PDD
        empirical_mean = np.mean(pairwise_density[:, time_slice_idx, :], axis=1)
        empirical_lower = np.quantile(
            pairwise_density[:, time_slice_idx, :], quantiles[0], axis=1
        )
        empirical_upper = np.quantile(
            pairwise_density[:, time_slice_idx, :], quantiles[1], axis=1
        )

        # Calculate mean and quantiles for CSR PDD
        csr_mean = np.mean(csr_pairwise_density[:, time_slice_idx, :], axis=1)
        csr_lower = np.quantile(
            csr_pairwise_density[:, time_slice_idx, :], quantiles[0], axis=1
        )
        csr_upper = np.quantile(
            csr_pairwise_density[:, time_slice_idx, :], quantiles[1], axis=1
        )

        # Create a line plot
        fig, axs = plt.subplots(figsize=(10, 5))

        # Plot mean and quantiles for empirical PDD
        axs.plot(support, empirical_mean, label="Empirical PDD")
        axs.fill_between(
            support,
            empirical_lower,
            empirical_upper,
            color="blue",
            alpha=0.2,
            label="Empirical CI",
        )

        # Plot mean and quantiles for CSR PDD
        axs.plot(csr_support, csr_mean, label="CSR PDD", color="orange")
        axs.fill_between(
            csr_support, csr_lower, csr_upper, color="orange", alpha=0.2, label="CSR CI"
        )

        # Add labels and title
        axs.set_xlabel("Distance")
        axs.set_ylabel("Density")
        axs.set_title(f"Pairwise Distance Density for Time Slice {time_slice}")

        # Add legend
        axs.legend()

        # Adjust layout and show plot
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4)
        return plt.gcf()


class TestPeakFinder:
    """
    Test peak finder
    """

    @pytest.fixture(name="grid_mesh")
    def grid_mesh_fixture(self, points_with_trend):
        """
        Fixture for grid mesh
        """
        # Define grid resolution and create the 2D grid for KDE evaluation
        # Get a bounding box for use later and to extract sensible distance limits
        x_min, y_min, x_max, y_max = get_box(points_with_trend)

        grid_resolution = 100
        x_grid = np.linspace(x_min, x_max, grid_resolution)
        y_grid = np.linspace(y_min, y_max, grid_resolution)
        return np.meshgrid(x_grid, y_grid)

    @pytest.fixture(name="grid")
    def grid_fixture(self, grid_mesh):
        """
        Fixture for grid
        """
        x_mesh, y_mesh = grid_mesh
        return np.vstack([x_mesh.ravel(), y_mesh.ravel()]).T

    @pytest.mark.mpl_image_compare(
        baseline_dir=IMG_DIR, filename="tutorial_peak_finder_points.png"
    )
    def test_peak_finder_points(self, points_with_trend, style_params):
        """
        Test peak finder points
        """
        ax, _ = chrono_plot(
            points_with_trend, style_params=style_params, time_slice=1500
        )
        ax.set_box_aspect(None, zoom=0.85)
        return ax.figure

    @pytest.fixture(name="kde_values")
    def kde_values_fixture(self, points_with_trend, grid, time_slices, grid_mesh):
        """
        Fixture for kde values
        """
        time_slice = time_slices[3]
        bandwidth = 1.5
        x_mesh, y_mesh = grid_mesh
        return kde_time(
            points=points_with_trend,
            time_slice=time_slice,
            bandwidth=bandwidth,
            grid=grid,
            output_shape=x_mesh.shape,
            kde_method=custom_kde,
        )

    @pytest.mark.parametrize("num_components, peak_finder", [(5, gmm_peak_finder)])
    @pytest.mark.mpl_image_compare(
        baseline_dir=IMG_DIR, filename="tutorial_peak_finder.png"
    )
    def test_peak_finder(
        self, points_with_trend, num_components, peak_finder, kde_values, grid_mesh
    ):
        """
        Test peak finder
        """
        peaks, weights = kde_peaks(
            points=points_with_trend, num_peaks=num_components, peak_finder=peak_finder
        )
        # Rank peaks by weight for interpretation
        ranked_indices = np.argsort(weights)[::-1]  # Sort weights in descending order
        ranked_peaks = peaks[ranked_indices]
        ranked_weights = weights[ranked_indices]

        # Print or plot ranked peaks for interpretation
        print("Ranked Peaks (from most to least important):")
        for i, (coord, weight) in enumerate(zip(ranked_peaks, ranked_weights), start=1):
            print(f"Rank {i}: Peak at {coord}, Weight = {weight:.4f}")

        # Plot KDE with ranked peaks labeled by importance
        x_mesh, y_mesh = grid_mesh
        plt.contourf(x_mesh, y_mesh, kde_values, levels=20, cmap="viridis")
        plt.scatter(
            ranked_peaks[:, 0],
            ranked_peaks[:, 1],
            color="red",
            marker="x",
            label="GMM Peaks",
        )
        for i, (x, y) in enumerate(ranked_peaks[:, :]):
            plt.text(x, y, f"Rank {i+1}", color="white", ha="center")
        plt.legend()
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("KDE with GMM Peaks Ranked by Importance")
        return plt.gcf()
