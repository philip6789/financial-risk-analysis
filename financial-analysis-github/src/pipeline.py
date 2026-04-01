

import numpy as np

from data_loader import (
    load_prices,
    compute_log_returns,
    build_portfolio_returns
)

from rolling_var import rolling_var_estimation
from backtesting import (
    compute_exceptions,
    exception_rate,
    kupiec_test,
    christoffersen_test,
    plot_rolling_var
)
from volatility import (ewma_volatility,ewma_var,rolling_garch_volatility,garch_var)
from bug_test import debug_xtp
from stress_testing import stress_testing_module


# 1. Data Preparation

tickers = ["AAPL", "MSFT", "GOOGL"]
weights = np.array([0.5, 0.3, 0.2])

prices = load_prices(
    tickers=tickers,
    start="2018-01-01",
    end="2023-12-31"
)

returns = compute_log_returns(prices)
portfolio_returns = build_portfolio_returns(returns, weights)



# 2. Rolling VaR Estimation and Backtesting

methods = ["historical", "parametric"]

results = {}

for method in methods:
    print(f"\n===== Backtesting ({method.upper()} VaR) =====")

    rolling_var = rolling_var_estimation(
        portfolio_returns,
        window=250,
        alpha=0.99,
        method=method
    )

    aligned_returns = portfolio_returns.loc[rolling_var.index]

    exceptions = compute_exceptions(
        aligned_returns,
        rolling_var
    )

    p_hat, x, t = exception_rate(exceptions)
    debug_xtp(x, t, p_hat)
    lr_uc, p_uc = kupiec_test(x, t, alpha=0.01)
    lr_cc, p_cc = christoffersen_test(exceptions)

    results[method] = {
        "exceptions": x,
        "exception_rate": p_hat,
        "kupiec_p": p_uc,
        "christoffersen_p": p_cc
    }

    print(f"Exceptions: {x}")
    print(f"Exception rate: {p_hat:.4f}")
    print(f"Kupiec p-value: {p_uc:.4f}")
    print(f"Christoffersen p-value: {p_cc:.4f}")

    plot_rolling_var(
        aligned_returns,
        rolling_var,
        exceptions,
        title=f"Rolling VaR Backtest ({method})"
    )


# 3. EWMA-based VaR
ewma_vol = ewma_volatility(
    returns=portfolio_returns,
    lambda_=0.94,
    window=250
)


ewma_var_99 = ewma_var(
    ewma_vol=ewma_vol,
    alpha=0.99
)
print(ewma_var_99.describe())
print((ewma_var_99 == 0).sum())
print((ewma_var_99.abs() < 1e-10).sum())


# ============ Align returns ============
aligned_returns = portfolio_returns.loc[ewma_var_99.index]

# ============ Backtesting ============
exceptions = compute_exceptions(aligned_returns, ewma_var_99)
p_hat, x, t = exception_rate(exceptions)
debug_xtp(x, t, p_hat)
lr_uc, p_uc = kupiec_test(x, t, alpha=0.01)
lr_ind, p_ind = christoffersen_test(exceptions)

# ============ Plot ============
plot_rolling_var(
    aligned_returns,
    ewma_var_99,
    exceptions,
    title="EWMA VaR (99%) Backtesting"
)

# 4. GARCH-based VaR
garch_vol = rolling_garch_volatility(
    returns=portfolio_returns,
    window=250,
    refit_every=5,
    p=1,
    q=1,
    dist="normal"
)

# ============ Convert to VaR ============
garch_var_99 = garch_var(
    garch_vol=garch_vol,
    alpha=0.99
)

# ============ Align returns ============
aligned_returns = portfolio_returns.loc[garch_var_99.index]

# ============ Backtesting ============
exceptions = compute_exceptions(aligned_returns, garch_var_99)

p_hat, x, t = exception_rate(exceptions)

lr_uc, p_uc = kupiec_test(x, t, alpha=0.01)

lr_ind, p_ind = christoffersen_test(exceptions)

print("====== Rolling GARCH VaR (99%) Backtesting ======")
print(f"Exception rate: {p_hat:.4f} ({x}/{t})")
print(f"Kupiec LR_uc: {lr_uc:.4f}, p-value: {p_uc:.4f}")
print(f"Christoffersen LR_ind: {lr_ind:.4f}, p-value: {p_ind:.4f}")

# ============ Plot ============
plot_rolling_var(
    aligned_returns,
    garch_var_99,
    exceptions,
    title="Rolling GARCH(1,1) VaR (99%) Backtesting"
)


# 5. Stress Testing and Scenario Analysis
print("\n========== Stress Testing & Scenario Analysis ==========")

stress_table = stress_testing_module(
    returns=portfolio_returns,
    portfolio_value=1_000_000,
    alpha_levels=(0.05, 0.01),
    worst_n_days=10,
    worst_percentile=0.01,
    hypothetical_shocks=(-0.10, -0.20, -0.30),
    vol_shock_multiplier=2.0
)
