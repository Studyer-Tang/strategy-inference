"""A standalone library example using simulated decimal returns."""

import json

import numpy as np

from strategy_inference import test_returns

rng = np.random.default_rng(17)
returns = rng.normal(0.0, 0.01, size=(512, 3))
returns[:, 0] += 0.0005
result = test_returns(returns, names=["carry", "value", "trend"],
    n_resamples=1999, seed=17, search_complete=True)
print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False))
