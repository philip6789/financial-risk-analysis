
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm

def ewma_volatility(
        returns: pd.Series,
        lambda_: float = 0.94,
        window: int = 250
):
    """
    Compute rolling EWMA (Exponentially Weighted Moving Average) volatility
    using RiskMetrics methodology.

    Parameters:
    ----------
    returns : pd.Series
        Time series of asset returns
    lambda_ : float
        Decay factor (typically 0.94 for daily data)
    window : int
        Initial window size for variance estimation

    Returns:
    -------
    pd.Series
        Time series of EWMA volatility
    """
    returns = returns.dropna()

    if window >= len(returns):
        raise ValueError("window too large")

    # 1. Initialize variance using sample variance over the initial window
    sigma2 = returns.iloc[:window].var()
    vol_series = []

    # 2. Recursively compute EWMA variance
    for t in range(window, len(returns)):
        r_prev = returns.iloc[t - 1]
        sigma2 = lambda_ * sigma2 + (1 - lambda_) * r_prev ** 2
        vol_series.append(np.sqrt(sigma2))

    vol_series = pd.Series(
        vol_series,
        index=returns.index[window:]
    )

    return vol_series

def ewma_var(
    ewma_vol: pd.Series,
    alpha: float = 0.99,
    mean: float = 0.0
):
    """
    Convert EWMA volatility to VaR
    """
    z = norm.ppf(1 - alpha)
    var_series = mean + ewma_vol * z
    return var_series




def rolling_garch_volatility(
    returns: pd.Series,
    window: int = 250,
    refit_every: int = 5,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    scale: float = 100.0
):
    """
    Rolling GARCH volatility forecasting (one-step ahead).

    Parameters
    ----------
    returns : pd.Series
        Daily returns series.
    window : int
        Rolling window size used for fitting.
    refit_every : int
        Refit GARCH model every N days to reduce computation.
    p, q : int
        GARCH(p, q) order.
    dist : str
        "normal" or "t".
    scale : float
        Scale factor for returns (arch library works better with % returns).

    Returns
    -------
    pd.Series
        One-step-ahead forecast volatility series (same index as rolling period).
    """

    returns = returns.dropna()

    if window >= len(returns):
        raise ValueError("window must be smaller than length of returns")

    vol_forecasts = []
    forecast_index = []

    last_fitted_model = None
    last_fit_result = None

    for t in range(window, len(returns)):
        sample = returns.iloc[t-window:t] * scale

        # Refit logic
        if (t == window) or ((t - window) % refit_every == 0):
            model = arch_model(
                sample,
                mean="Zero",      # risk management convention
                vol="GARCH",
                p=p,
                q=q,
                dist=dist
            )
            fit_result = model.fit(disp="off")
            last_fitted_model = model
            last_fit_result = fit_result

        # Forecast 1-step ahead volatility
        forecast = last_fit_result.forecast(horizon=1, reindex=False)
        sigma2_next = forecast.variance.values[-1, 0]
        sigma_next = np.sqrt(sigma2_next)

        # scale back to raw returns volatility
        sigma_next = sigma_next / scale

        vol_forecasts.append(sigma_next)
        forecast_index.append(returns.index[t])

    return pd.Series(vol_forecasts, index=forecast_index)

def garch_var(
    garch_vol: pd.Series,
    alpha: float = 0.99,
    mean: float = 0.0
):
    """
    Convert GARCH volatility forecast into VaR under Normal assumption.
    """
    z = norm.ppf(1 - alpha)
    return mean + garch_vol * z
