#!/usr/bin/env python 3.11.0
# -*-coding:utf-8 -*-
# @Author  : Christopher Carleton & Shuang Song
# @Contact   : carleton@gea.mpg.de
# GitHub   : https://github.com/wccarleton/chronocluster_dark

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.stats import rv_continuous
from scipy.stats.distributions import norm

# Get calibration curve
# Load the IntCal20 calibration curve
url = "https://intcal.org/curves/intcal20.14c"
intcal20 = pd.read_csv(url, skiprows=10, delimiter=",")
intcal20.columns = ["calbp", "c14bp", "c14_sigma", "f14c", "f14c_sigma"]


class calrcarbon:
    """Custom calibrated radiocarbon date distribution"""

    _interp_mean = None
    _interp_error = None

    def __init__(self, calcurve, c14_mean=None, c14_err=None):
        self.name = "calrcarbon"
        self.a = -max(calcurve["calbp"])
        self.b = -min(calcurve["calbp"])
        if calrcarbon._interp_mean is None:
            calrcarbon._interp_mean = CubicSpline(
                -calcurve["calbp"], -calcurve["c14bp"], extrapolate=False
            )
            calrcarbon._interp_error = CubicSpline(
                -calcurve["calbp"], calcurve["c14_sigma"], extrapolate=False
            )
        self.c14_mean = c14_mean
        self.c14_err = c14_err

    def _calc_curve_params(self, tau):
        curve_mean = calrcarbon._interp_mean(tau)
        curve_error = calrcarbon._interp_error(tau)
        return curve_mean, curve_error

    def _pdf(self, tau, c14_mean, c14_err):
        curve_mean, curve_error = self._calc_curve_params(tau)
        combined_error = np.sqrt(c14_err**2 + curve_error**2)
        return norm.pdf(c14_mean, loc=curve_mean, scale=combined_error)

    def _logpdf(self, tau, c14_mean, c14_err):
        curve_mean, curve_error = self._calc_curve_params(tau)
        combined_error = np.sqrt(c14_err**2 + curve_error**2)
        return norm.logpdf(c14_mean, loc=curve_mean, scale=combined_error)

    def _cdf(self, tau, c14_mean, c14_err):
        t_values, pdf_values = self._get_pdf_values(c14_mean, c14_err)
        cdf_values = np.cumsum(pdf_values) * (t_values[1] - t_values[0])
        cdf_values /= cdf_values[-1]
        return np.interp(tau, t_values, cdf_values)

    def _sf(self, tau, c14_mean, c14_err):
        return 1.0 - self._cdf(tau, c14_mean, c14_err)

    def _ppf(self, q, c14_mean, c14_err):
        t_values, pdf_values = self._get_pdf_values(c14_mean, c14_err)
        cdf_values = np.cumsum(pdf_values) * (t_values[1] - t_values[0])
        cdf_values /= cdf_values[-1]
        return np.interp(q, cdf_values, t_values)

    def _rvs(self, c14_mean, c14_err, size=None, random_state=None):
        if size is None:
            size = 1
        t_values, pdf_values = self._get_pdf_values(c14_mean, c14_err)
        cdf_values = np.cumsum(pdf_values) * (t_values[1] - t_values[0])
        cdf_values /= cdf_values[-1]
        uniform_samples = np.random.uniform(0, 1, size=size)
        inverse_cdf = np.interp(uniform_samples, cdf_values, t_values)
        return inverse_cdf

    def _get_pdf_values(self, c14_mean, c14_err, threshold=1e-7):
        t_values = np.linspace(self.a, self.b, 10000)
        pdf_values = self._pdf(t_values, c14_mean, c14_err)
        mask = pdf_values > threshold
        t_min = t_values[mask].min()
        t_max = t_values[mask].max()
        t_values = np.linspace(t_min, t_max, 10000)
        pdf_values = self._pdf(t_values, c14_mean, c14_err)
        pdf_values /= np.trapz(pdf_values, t_values)
        return t_values, pdf_values

    def pdf(self, tau, c14_mean=None, c14_err=None):
        """Public method for the PDF"""
        if c14_mean is None:
            c14_mean = self.c14_mean
        if c14_err is None:
            c14_err = self.c14_err
        return self._pdf(tau, c14_mean, c14_err)

    def logpdf(self, tau, c14_mean=None, c14_err=None):
        """Public method for the log PDF"""
        if c14_mean is None:
            c14_mean = self.c14_mean
        if c14_err is None:
            c14_err = self.c14_err
        return self._logpdf(tau, c14_mean, c14_err)

    def cdf(self, tau, c14_mean=None, c14_err=None):
        """Public method for the CDF"""
        if c14_mean is None:
            c14_mean = self.c14_mean
        if c14_err is None:
            c14_err = self.c14_err
        return self._cdf(tau, c14_mean, c14_err)

    def sf(self, tau, c14_mean=None, c14_err=None):
        """Public method for the survival function"""
        if c14_mean is None:
            c14_mean = self.c14_mean
        if c14_err is None:
            c14_err = self.c14_err
        return self._sf(tau, c14_mean, c14_err)

    def ppf(self, q, c14_mean=None, c14_err=None):
        """Public method for the PPF"""
        if c14_mean is None:
            c14_mean = self.c14_mean
        if c14_err is None:
            c14_err = self.c14_err
        return self._ppf(q, c14_mean, c14_err)

    def rvs(self, c14_mean=None, c14_err=None, size=None, random_state=None):
        """Public method for generating random variates"""
        if c14_mean is None:
            c14_mean = self.c14_mean
        if c14_err is None:
            c14_err = self.c14_err
        return self._rvs(c14_mean, c14_err, size=size, random_state=random_state)

    def mean(self, c14_mean=None, c14_err=None):
        """Public method for the mean of the distribution"""
        if c14_mean is None:
            c14_mean = self.c14_mean
        if c14_err is None:
            c14_err = self.c14_err
        t_values, pdf_values = self._get_pdf_values(c14_mean, c14_err)
        return np.sum(t_values * pdf_values) * (t_values[1] - t_values[0])

    def variance(self, c14_mean=None, c14_err=None):
        """Public method for the variance of the distribution"""
        if c14_mean is None:
            c14_mean = self.c14_mean
        if c14_err is None:
            c14_err = self.c14_err
        t_values, pdf_values = self._get_pdf_values(c14_mean, c14_err)
        mean_val = self.mean(c14_mean, c14_err)
        return np.sum((t_values - mean_val) ** 2 * pdf_values) * (
            t_values[1] - t_values[0]
        )

    def moment(self, n, c14_mean=None, c14_err=None):
        """Public method for the n-th moment of the distribution"""
        if c14_mean is None:
            c14_mean = self.c14_mean
        if c14_err is None:
            c14_err = self.c14_err
        t_values, pdf_values = self._get_pdf_values(c14_mean, c14_err)
        return np.sum(t_values**n * pdf_values) * (t_values[1] - t_values[0])


class ddelta_gen(rv_continuous):
    """Dirac Delta distribution"""

    def _argcheck(self, d):
        """Check the validity of the shape parameters"""
        return np.isfinite(d)

    def _pdf(self, x, d):
        """Probability density function"""
        x = np.asarray(x)
        return np.where(x == d, np.inf, 0.0)

    def _cdf(self, x, d):
        """Cumulative distribution function"""
        x = np.asarray(x)
        return np.where(x >= d, 1.0, 0.0)

    def _sf(self, x, d):
        """Survival function"""
        x = np.asarray(x)
        return np.where(x < d, 1.0, 0.0)

    def _ppf(self, q, d):
        """Percent point function (inverse of cdf)"""
        q = np.asarray(q)
        return np.full_like(q, d)

    def _rvs(self, d, size=None, random_state=None):
        """Random variates"""
        if size is None:
            size = 1
        return np.full(size, d)

    def mean(self, d):
        """Mean of the distribution"""
        return d

    def var(self, d):
        """Variance of the distribution"""
        return 0.0

    def std(self, d):
        """Standard deviation of the distribution"""
        return 0.0

    def _entropy(self, d):
        """Entropy of the distribution"""
        return 0.0

    def _moment(self, n, d):
        """nth moment of the distribution"""
        if n == 1:
            return d
        elif n == 2:
            return 0.0
        else:
            return np.nan


# Create the instance
ddelta = ddelta_gen(name="ddelta", shapes="d")
