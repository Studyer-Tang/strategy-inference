"""Simulated common-AR example; no historical returns or trading claim."""

import numpy as np

from strategy_inference import uncertainty_test, wilks_uncertainty_test
from strategy_inference.reference import gaussian_ar_mean_variance

rng = np.random.default_rng(20261119)
size, k, phi, rho = 512, 20, .9, .35
noise = np.sqrt(rho) * rng.standard_normal((size, 1)) + np.sqrt(1 - rho) * rng.standard_normal((size, k))
data = np.empty_like(noise)
data[0] = noise[0]
for time in range(1, size):
    data[time] = phi * data[time - 1] + np.sqrt(1 - phi**2) * noise[time]
data[:, 0] += 6 * np.sqrt(gaussian_ar_mean_variance(size, phi))

# Both use the same alpha, beta and GLS cutoff. Column order is fixed in advance.
for name, procedure in (("single-column F", uncertainty_test), ("joint multiscale", wilks_uncertainty_test)):
    result = procedure(data)
    print(name, "phi enclosure:", result.interval_bounds,
          "rejected columns:", np.flatnonzero(result.decisions).tolist())
