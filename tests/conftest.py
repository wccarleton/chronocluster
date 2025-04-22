#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Christopher Carleton & Shuang Song
# @Contact   : carleton@gea.mpg.de
# GitHub   : https://github.com/wccarleton/chronocluster_dark

import random

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.stats import norm

from chronocluster.calcurves import calibration_curves
from chronocluster.data.simdata import generate_random_points
from chronocluster.distributions import calrcarbon
from chronocluster.point import Point


@pytest.fixture(autouse=True)
def set_random_seed():
    """在每个测试前设置所有随机种子"""
    np.random.seed(42)
    random.seed(42)

    # 设置matplotlib的随机种子
    plt.rcParams["svg.hashsalt"] = "42"

    yield

    # 测试后清理
    plt.close("all")


@pytest.fixture(autouse=True)
def mpl_settings():
    """固定matplotlib的设置"""
    with plt.style.context("default"):
        # 固定DPI和图像大小
        plt.rcParams["figure.dpi"] = 100
        plt.rcParams["figure.figsize"] = [6.4, 4.8]
        # 禁用动画
        plt.rcParams["animation.html"] = "none"
        # 使用固定的后端
        plt.rcParams["backend"] = "agg"
        yield


@pytest.fixture(name="dist1")
def dist1():
    """
    Fixture for a normal distribution with mean 100 and standard deviation 10.
    """
    return norm(loc=100, scale=10)


@pytest.fixture(name="dist2")
def dist2():
    """
    Fixture for a normal distribution with mean 200 and standard deviation 10.
    """
    return norm(loc=200, scale=10)


@pytest.fixture(name="point")
def point_fixture():
    """
    Fixture for a list of Point objects.
    """
    # define cluster distributions
    cluster_center = (10, 10)
    cluster_std = 1.0

    # randomly select point coords
    x_coord = np.random.normal(loc=cluster_center[0], scale=cluster_std, size=1)[0]
    y_coord = np.random.normal(loc=cluster_center[1], scale=cluster_std, size=1)[0]

    start_age_mean = 1000
    start_age_err = 50
    end_age_mean = 1200
    end_age_err = 50

    # create random start and end ages with uncertainty
    start_dist = norm(loc=start_age_mean, scale=start_age_err)
    end_dist = norm(loc=end_age_mean, scale=end_age_err)

    # finally, generate random point
    point = Point(
        x=x_coord, y=y_coord, start_distribution=start_dist, end_distribution=end_dist
    )
    return [point]


@pytest.fixture(name="curve")
def curve_fixture():
    """
    Fixture for a custom distribution.
    """
    calcurve = calibration_curves["intcal20"]
    return calrcarbon(calcurve, c14_mean=-5000, c14_err=20)


@pytest.fixture(name="points")
def points_fixture():
    """
    Fixture for a list of Point objects.
    """
    # define cluster distributions
    cluster_centers = [(10, 10), (20, 20)]
    cluster_stds = [1.0, 1.0]

    # create random start and end ages with uncertainty for each point
    # define dating models and uncertainties
    start_type = "norm"
    start_hyperparams = [1000, 50, 100]
    end_type = "constant"
    end_hyperparams = [1700]

    # finally, generate 100 random points using the above models
    points = generate_random_points(
        50,
        cluster_centers,
        cluster_stds,
        start_type,
        start_hyperparams,
        end_type,
        end_hyperparams,
    )
    return points


@pytest.fixture(name="time_slices")
def time_slices_fixture():
    """
    Fixture for a time slice.
    """
    # Define the time slices
    start_time = 1000
    end_time = 1700
    time_interval = 50
    return np.arange(start_time, end_time, time_interval)


@pytest.fixture(name="style_params")
def style_params_fixture():
    """
    Fixture for style parameters.
    This is the style parameters from the tutorial notebook.
    """
    return {
        "start_mean_color": None,  # Do not plot start mean points
        "end_mean_color": None,  # Do not plot end mean points
        "mean_point_size": 10,
        "cylinder_color": (0.3, 0.3, 0.3),  # Dark grey
        "ppf_limits": (0.05, 0.95),  # Use different ppf limits
        "shadow_color": (0.4, 0.4, 0.4),  # grey
        "shadow_size": 10,
        "time_slice_color": (0.5, 0.5, 0.5),  # Grey
        "time_slice_alpha": 0.3,
        "time_slice_point_color": (0, 0, 0),  # Black
    }


@pytest.fixture(name="points_with_trend")
def points_with_trend_fixture():
    """
    Fixture for a list of Point objects with a linear trend across the landscape.
    """
    # Define cluster distributions, but with a linear trend across the landscape (e.g., river valley)
    cluster_centers = [(100, 100), (110, 110), (120, 120)]
    cluster_stds = [0.5, 0.5, 0.5]

    # Define parameters for the temporal distributions
    start_age_mean = 1000
    start_age_err = 50
    end_age_mean = 1700
    end_age_err = 10

    # Define a directional trend to simulate a river valley
    num_points_per_cluster = 50

    # Generate points with clusters along a directional trend
    points = []
    for center, std in zip(cluster_centers, cluster_stds):
        # Generate base points for this cluster
        x_coords = np.random.normal(center[0], std, num_points_per_cluster)
        y_coords = np.random.normal(center[1], std, num_points_per_cluster)

        for x, y in zip(x_coords, y_coords):
            # Create start and end distributions for each point
            start_dist = norm(loc=start_age_mean, scale=start_age_err)
            end_dist = norm(loc=end_age_mean, scale=end_age_err)

            # Append the point with spatial and temporal distributions
            point = Point(
                x=x, y=y, start_distribution=start_dist, end_distribution=end_dist
            )
            points.append(point)
    return points
