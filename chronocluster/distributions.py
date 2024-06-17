import numpy as np
import pandas as pd
from scipy.stats import rv_continuous
from scipy.stats.distributions import norm
from scipy.interpolate import interp1d, CubicSpline

# Get calibration curve
# Load the IntCal20 calibration curve
url = "https://intcal.org/curves/intcal20.14c"
intcal20 = pd.read_csv(url, skiprows=10, delimiter=",")
intcal20.columns = ["calbp", "c14bp", "c14_sigma", "f14c", "f14c_sigma"]

class calrcarbon(rv_continuous):
    """Probability functions for radiocarbon dates"""
    _interp_mean = None
    _interp_error = None

    def __init__(self, calcurve):
        super().__init__(name='calrcarbon', shapes='c14_mean, c14_err')
        self.dist = self
        self.badvalue = np.nan
        self.xtol = 1e-14
        self.a = -max(calcurve['calbp'])
        self.b = -min(calcurve['calbp'])
        if calrcarbon._interp_mean is None:
            calrcarbon._interp_mean = CubicSpline(-calcurve['calbp'], -calcurve['c14bp'], extrapolate=False)
            calrcarbon._interp_error = CubicSpline(-calcurve['calbp'], calcurve['c14_sigma'], extrapolate=False)

    def _argcheck(self, c14_mean, c14_err):
        return (c14_err > 0) & np.isfinite(c14_mean)

    def _calc_curve_params(self, tau):
        curve_mean, curve_error = calrcarbon._interp_mean(tau), calrcarbon._interp_error(tau)
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

    def mean(self, c14_mean, c14_err):
        t_values, pdf_values = self._get_pdf_values(c14_mean, c14_err)
        return np.sum(t_values * pdf_values) * (t_values[1] - t_values[0])

    def variance(self, c14_mean, c14_err):
        t_values, pdf_values = self._get_pdf_values(c14_mean, c14_err)
        mean_val = self.mean(c14_mean, c14_err)
        return np.sum((t_values - mean_val) ** 2 * pdf_values) * (t_values[1] - t_values[0])

    def _munp(self, n, c14_mean, c14_err):
        """n-th moment of the distribution"""
        t_values, pdf_values = self._get_pdf_values(c14_mean, c14_err)
        return np.sum(t_values**n * pdf_values) * (t_values[1] - t_values[0])

class ddelta(rv_continuous):
    """Probability functions approximating the Dirac Delta"""
    
    def __init__(self, d):
        super().__init__(name='ddelta', shapes='d')
        self.d = d
        self.dist = self
        self.badvalue = np.nan
        self.a = d
        self.b = d
        self.xtol = 1e-14
        self.moment_type = 1
        self.shapes = None
        self.numargs = 0
        self.vecentropy = np.vectorize(self._entropy)
        self.generic_moment = np.vectorize(self._moment)

    def _argcheck(self, d):
        """Check the validity of the shape parameters"""
        return np.isfinite(d)

    def _pdf(self, x):
        """Probability density function"""
        return np.inf if x == self.d else 0

    def _cdf(self, x):
        """Cumulative distribution function"""
        return 1.0 if x >= self.d else 0.0
    
    def _sf(self, x):
        """Survival function"""
        return 1.0 - self._cdf(x)

    def _ppf(self, q):
        """Percent point function (inverse of cdf)"""
        return self.d

    def _rvs(self, size=None, random_state=None):
        """Random variates"""
        return np.full(size, self.d)

    def mean(self):
        """Mean of the distribution"""
        return self.d
    
    def var(self):
        """Variance of the distribution"""
        return 0.0

    def std(self):
        """Standard deviation of the distribution"""
        return 0.0

    def _entropy(self, *args, **kwargs):
        """Entropy of the distribution"""
        return 0.0

    def _moment(self, n, *args, **kwargs):
        """nth moment of the distribution"""
        if n == 1:
            return self.mean()
        elif n == 2:
            return self.var()**2
        else:
            return np.nan