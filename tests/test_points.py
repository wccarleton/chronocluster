#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Christopher Carleton & Shuang Song
# @Contact   : carleton@gea.mpg.de
# GitHub   : https://github.com/wccarleton/chronocluster

"""
Tests for the Point class
-------------------

This module contains tests for the Point class functionality.
"""

import warnings

import numpy as np
import pytest
from scipy.stats import norm

from chronocluster.point import Point


class TestPointInitialization:
    """
    Test basic initialization of Point class
    -------

    Verifies that a Point object can be correctly initialized with valid inputs
    and that all attributes are properly set.
    """

    def test_point_initialization(self, dist1, dist2):
        """
        Test basic initialization of Point class
        -----------------------------------

        Verifies that a Point object can be correctly initialized with valid inputs
        and that all attributes are properly set.
        """
        # Arrange
        point = Point(1.0, 2.0, dist1, dist2)

        # Act / Assert
        assert point.x == 1.0
        assert point.y == 2.0
        assert point.start_distribution == dist1
        assert point.end_distribution == dist2

    @pytest.mark.parametrize(
        "x, y",
        [
            ("1.0", 2.0),  # string is not a valid coordinate
            (1.0, np.nan),  # NaN is not a valid coordinate
        ],
    )
    def test_invalid_coordinates(self, x, y, dist1, dist2):
        """
        Test invalid coordinate inputs
        --------------------------

        Verifies that the Point class correctly raises TypeError for invalid
        coordinate inputs.
        """
        with pytest.raises((TypeError, ValueError)):
            Point(x, y, dist1, dist2)

    @pytest.mark.parametrize(
        "loc1, loc2, scale1, scale2",
        [
            (100, 110, 20, 20),
            (10, 11, 5, 5),
        ],
    )
    def test_temporal_consistency_warning(self, loc1, loc2, scale1, scale2):
        """
        Test temporal consistency warnings
        ------------------------------

        Verifies that appropriate warnings are raised when temporal distributions
        have significant overlap.
        """
        # Arrange
        start_dist = norm(loc=loc1, scale=scale1)
        end_dist = norm(loc=loc2, scale=scale2)

        # Act / Assert
        with warnings.catch_warnings(record=True) as w:
            Point(1.0, 2.0, start_dist, end_dist)
            assert len(w) > 0
            assert "Significant overlap" in str(w[-1].message)


class TestPointMethods:
    """
    Test methods of the Point class
    ------------------------------

    Verifies that the methods of the Point class work as expected.
    """

    @pytest.mark.parametrize(
        "x, y, dist, expected_repr",
        [
            (
                1,
                2,
                norm(loc=100, scale=10),
                "Point(x=1, y=2, start_distribution=norm(loc=100.0, scale=10.0), end_distribution=norm(loc=200.0, scale=10.0))",
            ),
            # TODO: Add ddelta test
            # TODO: Add calrcarbon test
        ],
    )
    def test_repr(self, x, y, dist, dist2, expected_repr):
        """
        Test string representation of Point class
        ----------------------------------------

        Verifies that the string representation of the Point class is correct.
        """
        # Arrange
        point = Point(x, y, dist, dist2)

        # Act / Assert
        assert repr(point) == expected_repr

    @pytest.mark.parametrize(
        "time_slice, expected_prob",
        [
            (150, 1),
            (50, 0.0),
            (250, 0.0),
        ],
    )
    def test_calculate_inclusion_probability(
        self,
        dist1,
        dist2,
        time_slice,
        expected_prob,
    ):
        """
        Test inclusion probability calculation
        ---------------------------------

        Verifies that inclusion probabilities are correctly calculated for various
        time slices.
        """
        # Arrange
        point = Point(1.0, 2.0, dist1, dist2)

        # Act
        prob = point.calculate_inclusion_probability(time_slice)

        # Assert
        assert np.isclose(prob, expected_prob, atol=1e-3)

    @pytest.mark.parametrize(
        "time_slice, expected_prob",
        [
            (np.array([150, 50, 250]), np.array([1, 0, 0])),
        ],
    )
    def test_calculate_inclusion_probability_array(
        self, dist1, dist2, time_slice, expected_prob
    ):
        """
        Test inclusion probability calculation for an array of time slices
        --------------------------------------------------------------

        Verifies that inclusion probabilities are correctly calculated for an array of time slices.
        """
        point = Point(1.0, 2.0, dist1, dist2)
        prob = point.calculate_inclusion_probability(time_slice)
        assert np.allclose(prob, expected_prob, atol=1e-3)

    @pytest.mark.parametrize(
        "start_dist, end_dist, expected_overlap",
        [
            (norm(loc=100, scale=10), norm(loc=200, scale=10), 0.0),
            (norm(loc=100, scale=10), norm(loc=100, scale=10), 0.5),
            (norm(loc=100, scale=10), norm(loc=110, scale=10), 1 / 3),
        ],
    )
    def test_calculate_overlap_ratio(self, start_dist, end_dist, expected_overlap):
        """
        Test overlap ratio calculation
        --------------------------

        Verifies that overlap ratios are correctly calculated for different
        temporal distribution configurations.
        """
        # Arrange
        point = Point(1.0, 2.0, start_dist, end_dist, verbose=False)

        # Act
        overlap = getattr(point, "_calculate_overlap_ratio")()

        # Assert
        assert np.isclose(overlap, expected_overlap, atol=1e-1)

    def test_to_dict(self, dist1, dist2):
        """
        Test to_dict method
        -----------------

        Verifies that the to_dict method returns a dictionary with the correct keys and values.
        """
        point = Point(1.0, 2.0, dist1, dist2)
        assert point.to_dict() == {
            "x": 1.0,
            "y": 2.0,
            "start_type": dist1.dist.name,
            "start_params": [dist1.mean(), dist1.std()],
            "end_type": dist2.dist.name,
            "end_params": [dist2.mean(), dist2.std()],
        }
