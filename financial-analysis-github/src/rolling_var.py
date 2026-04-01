
import pandas as pd
import numpy as np
from scipy.stats import norm

def rolling_var_estimation(returns, window=250, alpha=0.99, method="parametric"):
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)

    if window >= len(returns):
        raise ValueError("window size must be smaller than length of returns")

    rolling_var = []

    for t in range(window, len(returns)):
        sample = returns[t - window:t]

        if method == "historical":
            var_t = np.quantile(sample, 1 - alpha)

        elif method == "parametric":
            mu = sample.mean()
            sigma = sample.std(ddof=1)
            var_t = mu + sigma * norm.ppf(1 - alpha)

        else:
            raise ValueError("method must be 'historical' or 'parametric'")

        rolling_var.append(var_t)

    rolling_var = pd.Series(rolling_var, index=returns.index[window:])
    return rolling_var
