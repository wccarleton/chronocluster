import numpy as np
import pandas as pd
from scipy.stats import rv_continuous, norm
from scipy.interpolate import interp1d, CubicSpline
from scipy.integrate import quad
from statsmodels.distributions.empirical_distribution import ECDF
from chronocluster.distributions import calrcarbon, ddelta

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Get calibration curve
# Load the IntCal20 calibration curve
url = "https://intcal.org/curves/intcal20.14c"
intcal20 = pd.read_csv(url, skiprows=10, delimiter=",")
intcal20.columns = ["calbp", "c14bp", "c14_sigma", "f14c", "f14c_sigma"]

# Example of calibration curve data
print(intcal20.head())

calcurve = intcal20

# Precompute the splines
interp_mean = CubicSpline(-calcurve['calbp'], -calcurve['c14bp'], extrapolate=False)
interp_error = CubicSpline(-calcurve['calbp'], calcurve['c14_sigma'], extrapolate=False)

# Parameters
c14_mean = -4500  # Example mean radiocarbon measurement (negative BP)
c14_err = 20
n_samples = 10000

# Instantiate the calrcarbon distribution with the calibration curve
cal_rc = calrcarbon(calcurve)
cal_rc.pdf(-8000, c14_mean, c14_err)

mynorm = norm(loc=10,scale=2)

norm.pdf(10,loc=10)

# 1. Calibrate the date using the pdf function and plot it
t_values, pdf_values = cal_rc._get_pdf_values(c14_mean=c14_mean, c14_err=c14_err)

plt.figure(figsize=(10, 6))
plt.plot(t_values, pdf_values, label='Calibrated PDF', color='orange')
plt.xlabel('Calendar Date (cal BP)')
plt.ylabel('Density')
plt.title('Calibrated Radiocarbon Date PDF')
plt.legend()
plt.show()

# 2. Generate radiocarbon samples and plot the histogram with the PDF for comparison
c14_samples = cal_rc._rvs(c14_mean=c14_mean, c14_err=c14_err, size=n_samples)

plt.figure(figsize=(10, 6))
plt.hist(c14_samples, bins=100, density=True, alpha=0.5, color='blue', label='Sample Histogram')
plt.plot(t_values, pdf_values, color='orange', linewidth=2, label='Calculated PDF')
plt.xlabel('Calendar Date (cal BP)')
plt.ylabel('Density')
plt.title('Histogram of Calibrated Radiocarbon Date Samples')
plt.legend()
plt.show()

# 3. Compute empirical CDF
ecdf = ECDF(c14_samples)

# Generate CDF values for the calculated CDF
cal_dates = np.arange(t_values.min(), t_values.max(), 0.25)  # Use the range of t_values
calculated_cdf_values = cal_rc._cdf(cal_dates, c14_mean=c14_mean, c14_err=c14_err)

# Generate empirical CDF values for the same range of calendar dates
empirical_cdf_values = ecdf(cal_dates)

# Plot the calculated CDF
plt.figure(figsize=(10, 6))
plt.plot(cal_dates, calculated_cdf_values, label='Calculated CDF', color='orange', linewidth=10)

# Plot the empirical CDF
plt.step(cal_dates, empirical_cdf_values, label='Empirical CDF', color='blue', where='post', linewidth=2)

# Formatting the plot
plt.xlabel('Calendar Date (cal BP)')
plt.ylabel('Cumulative Probability')
plt.title('Comparison of Calculated and Empirical CDFs')
plt.legend()
plt.show()

# Check CDF properties
is_monotonic = np.all(np.diff(calculated_cdf_values) >= 0)
print(f"CDF is monotonic: {is_monotonic}")

cdf_at_min = np.isclose(calculated_cdf_values[0], 0, atol=1e-6)
cdf_at_max = np.isclose(calculated_cdf_values[-1], 1, atol=1e-6)
print(f"CDF approaches 0 at the lower bound: {cdf_at_min}")
print(f"CDF approaches 1 at the upper bound: {cdf_at_max}")