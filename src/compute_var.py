import numpy as np
from scipy.stats import norm

def compute_var(returns, method="historical", confidence=0.99, simulations=100000):
    """
    Unified interface for VaR computation.

    Parameters:
    returns : pd.Series or np.array
    method  : "historical", "parametric", "mc"
    confidence : float
    simulations : int (only used for Monte Carlo)

    Returns:
    var_value : float
    """

    if method == "historical":
        return np.percentile(returns, (1 - confidence) * 100)

    elif method == "parametric":
        mu = returns.mean()
        sigma = returns.std()
        return norm.ppf(1 - confidence, mu, sigma)

    elif method == "mc":
        mu = returns.mean()
        sigma = returns.std()
        sim = np.random.normal(mu, sigma, simulations)
        return np.percentile(sim, (1 - confidence) * 100)

    else:
        raise ValueError(f"Unknown VaR method: {method}")
