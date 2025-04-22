#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Christopher Carleton & Shuang Song
# @Contact   : carleton@gea.mpg.de
# GitHub   : https://github.com/wccarleton/chronocluster_dark

import warnings
from dataclasses import dataclass
from typing import Any, TypeAlias, Union

import numpy as np
from scipy.integrate import simpson
from scipy.stats import norm, rv_continuous

NumType: TypeAlias = float | int


# Using dataclass to define the Point class,
# which is a data structure to store the spatial and temporal information of a point.
# The dataclass is a built-in class in Python that is used to store data.
# It's more convenient and efficient.
@dataclass
class Point:
    """
    A class to represent a point with spatial coordinates and temporal distributions.

    Attributes:
    -----------
    x : float
        The x-coordinate of the point.
    y : float
        The y-coordinate of the point.
    start_distribution : scipy.stats.rv_continuous
        The probability distribution for the start date.
    end_distribution : scipy.stats.rv_continuous
        The probability distribution for the end date.
    verbose : bool, optional
        If True, warnings will be shown. Default is False.

    Methods:
    --------
    __init__(self, x, y, start_distribution, end_distribution, verbose=False):
        Initializes a new Point instance with the given coordinates and distributions.

    _check_distributions(self):
        Checks the temporal consistency of the start and end distributions.

    _calculate_overlap_ratio(self):
        Calculates the overlap ratio between the start and end distributions.

    calculate_inclusion_probability(self, time_slice):
        Calculates the inclusion probability of the point for a given time slice.

    _repr_distribution(self, dist):
        Returns a string representation of the distribution.

    __repr__(self):
        Returns a string representation of the Point instance.
    """

    x: NumType
    y: NumType
    start_distribution: rv_continuous
    end_distribution: rv_continuous
    verbose: bool = True

    def __post_init__(self):
        if not isinstance(self.x, (int, float)):
            raise TypeError("x must be a scalar (int or float)")
        if not isinstance(self.y, (int, float)):
            raise TypeError("y must be a scalar (int or float)")
        if np.isnan(self.x) or np.isnan(self.y):
            raise ValueError("x and y must be valid numbers")
        # Perform checks
        self._check_distributions()

    def _check_distributions(self):
        """
        Checks the temporal consistency of the start and end distributions.
        Warnings are only shown if verbose=True.
        """
        overlap_ratio = self._calculate_overlap_ratio()
        if overlap_ratio > 0.25 and self.verbose:
            warnings.warn(
                f"Significant overlap between start and end "
                f"distributions. Overlap ratio: {overlap_ratio:.2f}"
            )

        start_mean = self.start_distribution.mean()
        end_mean = self.end_distribution.mean()
        if end_mean < start_mean and self.verbose:
            warnings.warn(
                f"End date distribution mean ({end_mean}) is earlier "
                f"than start date distribution mean ({start_mean}). Possible "
                f"data error."
            )

    def _calculate_overlap_ratio(
        self,
        min_prob: float = 0.01,
        max_prob: float = 0.99,
    ) -> float:
        """
        Calculates the overlap ratio between the start and end distributions.

        Parameters:
        -----------
        min_prob : float, optional
            The minimum probability to use for the integration. Default is 0.01.
        max_prob : float, optional
            The maximum probability to use for the integration. Default is 0.99.

        Raises:
        --------
        ValueError: If min_prob is greater than max_prob or any is not between 0 and 1.

        Returns:
        --------
        float
            The overlap ratio between the start and end distributions.
        """
        if min_prob > max_prob:
            raise ValueError("min_prob must be less than max_prob.")
        if min_prob < 0 or max_prob > 1:
            raise ValueError("min_prob and max_prob must be between 0 and 1.")
        # Define a reasonable range for integration
        range_min = min(
            self.start_distribution.ppf(min_prob),
            self.end_distribution.ppf(min_prob),
        )
        range_max = max(
            self.start_distribution.ppf(max_prob),
            self.end_distribution.ppf(max_prob),
        )

        # Generate a dense range of values for the PDFs
        x = np.linspace(range_min, range_max, 1000)

        # Compute the PDF values
        start_pdf = self.start_distribution.pdf(x)
        end_pdf = self.end_distribution.pdf(x)

        # Calculate the overlap area using the minimum of the two PDFs
        overlap_pdf = np.minimum(start_pdf, end_pdf)
        overlap_area = simpson(overlap_pdf, x=x)

        # Calculate the total area of the two distributions
        total_area_start = simpson(start_pdf, x=x)
        total_area_end = simpson(end_pdf, x=x)

        # Calculate combined area
        combined_area = total_area_start + total_area_end

        # Calculate the overlap ratio
        if combined_area == 0:
            warnings.warn(
                "Sum of density integrals is zero! Check that start "
                "and end dates are present and, if constant, not identical."
            )
            overlap_ratio = np.nan
        else:
            overlap_ratio = overlap_area / combined_area

        return overlap_ratio

    def calculate_inclusion_probability(
        self, time_slice: Union[NumType, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculates the inclusion probability of the point for given time slice(s).

        Parameters:
        -----------
        time_slice : float or numpy.ndarray
            The time slice(s) to calculate the inclusion probability for.
            Can be a single value or an array of values.

        Returns:
        --------
        float or numpy.ndarray
            The inclusion probability for the given time slice(s).
            Returns a single float if input is a scalar, or an array if input is an array.
        """
        start_prob = self.start_distribution.cdf(time_slice)
        end_prob = self.end_distribution.sf(time_slice)

        # Handle scalar and array cases
        if isinstance(time_slice, np.ndarray):
            return np.where(start_prob <= 0, 0.0, start_prob * end_prob)
        return 0.0 if start_prob <= 0 else start_prob * end_prob

    def _repr_distribution(self, dist):
        """
        Returns a string representation of the distribution.

        Parameters:
        -----------
        dist : scipy.stats.rv_continuous
            The distribution to represent as a string.

        Returns:
        --------
        str
            The string representation of the distribution.
        """
        if hasattr(dist, "name") and dist.name == "calrcarbon":
            return f"calrcarbon(c14_mean={dist.c14_mean}, c14_err={dist.c14_err})"
        if dist.dist.name == "norm":
            loc = dist.mean()
            scale = dist.std()
            return f"norm(loc={loc}, scale={scale})"
        if dist.dist.name == "ddelta":
            d = dist.mean()
            return f"ddelta(d={d})"
        params = {
            key: value
            for key, value in dist.__dict__.items()
            if not key.startswith("_")
        }
        param_str = ", ".join([f"{key}={value}" for key, value in params.items()])
        return f"{dist.dist.name}({param_str})"

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        """
        Returns a string representation of the Point instance.

        Returns:
        --------
        str
            The string representation of the Point instance.
        """
        start_repr = self._repr_distribution(self.start_distribution)
        end_repr = self._repr_distribution(self.end_distribution)
        return (
            f"Point(x={self.x}, y={self.y}, "
            f"start_distribution={start_repr}, "
            f"end_distribution={end_repr})"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Point object to a dictionary.
        """
        start_type = self.start_distribution.dist.name
        start_params = [self.start_distribution.mean(), self.start_distribution.std()]
        end_type = self.end_distribution.dist.name
        end_params = [self.end_distribution.mean(), self.end_distribution.std()]
        return {
            "x": self.x,
            "y": self.y,
            "start_type": start_type,
            "start_params": start_params,
            "end_type": end_type,
            "end_params": end_params,
        }


def generate_norm_point(
    coordinates: tuple[NumType, NumType],
    start_dist: tuple[NumType, NumType],
    end_dist: tuple[NumType, NumType],
) -> Point:
    """
    Generate a Point object with normal distributions for start and end dates.

    Parameters:
    -----------
    coordinates: tuple[NumType, NumType]
        The coordinates of the point.
    start_dist: tuple[NumType, NumType]
        The mean and standard deviation of the start distribution.
    end_dist: tuple[NumType, NumType]
        The mean and standard deviation of the end distribution.
    """
    x, y = coordinates
    start_dist = norm(loc=start_dist[0], scale=start_dist[1])
    end_dist = norm(loc=end_dist[0], scale=end_dist[1])
    return Point(x, y, start_dist, end_dist)
