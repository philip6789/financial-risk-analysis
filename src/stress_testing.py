import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def compute_var_es(returns, alpha=0.05, method="historical"):
    """
    Compute VaR and Expected Shortfall (ES) based on returns series.

    Parameters:
        returns: pd.Series
        alpha: float (e.g. 0.05 for 95% VaR)
        method: "historical" or "parametric"

    Returns:
        var_value: float
        es_value: float
    """

    if method == "historical":
        var_value = np.quantile(returns, alpha)
        tail_losses = returns[returns <= var_value]

        if len(tail_losses) == 0:
            es_value = np.nan
        else:
            es_value = tail_losses.mean()

        return var_value, es_value

    elif method == "parametric":
        mu = returns.mean()
        sigma = returns.std()
        var_value = mu + sigma * norm.ppf(alpha)

        # ES for normal distribution
        es_value = mu - sigma * norm.pdf(norm.ppf(alpha)) / alpha

        return var_value, es_value

    else:
        raise ValueError("method must be 'historical' or 'parametric'")


def stress_testing_module(
    returns,
    portfolio_value=1_000_000,
    alpha_levels=(0.05, 0.01),
    worst_n_days=10,
    worst_percentile=0.01,
    hypothetical_shocks=(-0.10, -0.20, -0.30),
    vol_shock_multiplier=2.0,
):
    """
    Stress Testing & Scenario Analysis Module

    Parameters:
        returns: pd.Series (daily returns)
        portfolio_value: float
        alpha_levels: tuple (e.g. (0.05, 0.01) for 95% and 99%)
        worst_n_days: int
        worst_percentile: float
        hypothetical_shocks: tuple of floats
        vol_shock_multiplier: float

    Returns:
        stress_summary_df: pd.DataFrame
    """

    returns = returns.dropna().copy()

    # ==============================
    # 1) Historical Stress Testing
    # ==============================
    sorted_returns = returns.sort_values()

    worst_n = sorted_returns.head(worst_n_days)
    worst_pct_threshold = np.quantile(returns, worst_percentile)
    worst_pct_days = returns[returns <= worst_pct_threshold]

    historical_stress_results = {
        "Worst N Days (Mean Return)": worst_n.mean(),
        "Worst N Days (Min Return)": worst_n.min(),
        f"Worst {int(worst_percentile*100)}% Days (Mean Return)": worst_pct_days.mean(),
        f"Worst {int(worst_percentile*100)}% Days (Min Return)": worst_pct_days.min(),
    }

    # Convert to monetary loss
    historical_stress_loss = {
        k.replace("Return", "Loss"): -v * portfolio_value
        for k, v in historical_stress_results.items()
    }

    # ==============================
    # 2) Hypothetical Scenarios
    # ==============================
    scenario_results = {}
    for shock in hypothetical_shocks:
        scenario_name = f"Hypothetical Shock {int(shock*100)}%"
        scenario_results[scenario_name] = shock

    scenario_loss = {k: -v * portfolio_value for k, v in scenario_results.items()}

    # ==============================
    # 3) Stress VaR (Volatility Shock)
    # ==============================
    stress_var_results = {}
    stress_es_results = {}

    mu = returns.mean()
    sigma = returns.std()
    stressed_sigma = vol_shock_multiplier * sigma

    for alpha in alpha_levels:
        z = norm.ppf(alpha)
        stressed_var = mu + stressed_sigma * z
        stressed_es = mu - stressed_sigma * norm.pdf(z) / alpha

        stress_var_results[f"Stress VaR ({int((1-alpha)*100)}%)"] = stressed_var
        stress_es_results[f"Stress ES ({int((1-alpha)*100)}%)"] = stressed_es

    stress_var_loss = {k: -v * portfolio_value for k, v in stress_var_results.items()}
    stress_es_loss = {k: -v * portfolio_value for k, v in stress_es_results.items()}

    # ==============================
    # 4) Expected Shortfall (Historical & Parametric)
    # ==============================
    var_es_results = {}

    for alpha in alpha_levels:
        var_hist, es_hist = compute_var_es(returns, alpha=alpha, method="historical")
        var_para, es_para = compute_var_es(returns, alpha=alpha, method="parametric")

        var_es_results[f"Historical VaR ({int((1-alpha)*100)}%)"] = var_hist
        var_es_results[f"Historical ES ({int((1-alpha)*100)}%)"] = es_hist
        var_es_results[f"Parametric VaR ({int((1-alpha)*100)}%)"] = var_para
        var_es_results[f"Parametric ES ({int((1-alpha)*100)}%)"] = es_para

    var_es_loss = {k.replace("VaR", "Loss").replace("ES", "Loss"): -v * portfolio_value
                   for k, v in var_es_results.items()}

    # ==============================
    # 5) Build Summary Table
    # ==============================
    summary_dict = {}

    # Historical Stress
    summary_dict.update(historical_stress_loss)

    # Hypothetical Stress
    summary_dict.update(scenario_loss)

    # Stress VaR / ES
    summary_dict.update(stress_var_loss)
    summary_dict.update(stress_es_loss)

    # Normal VaR / ES (Historical + Parametric)
    summary_dict.update(var_es_loss)

    stress_summary_df = pd.DataFrame.from_dict(summary_dict, orient="index", columns=["Loss (Currency)"])
    stress_summary_df["Loss (Currency)"] = stress_summary_df["Loss (Currency)"].astype(float)

    # ==============================
    # 6) Visualization (Bar Chart)
    # ==============================
    plt.figure(figsize=(12, 6))
    stress_summary_df.sort_values("Loss (Currency)", ascending=False).plot(kind="bar", legend=False)
    plt.title("Stress Testing Summary - Loss under Different Scenarios")
    plt.ylabel("Loss")
    plt.xlabel("Scenario")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # ==============================
    # 7) Visualization (Worst Days Time Series Highlight)
    # ==============================
    plt.figure(figsize=(12, 5))
    plt.plot(returns.index, returns, label="Daily Returns", alpha=0.6)

    plt.scatter(worst_n.index, worst_n.values, label=f"Worst {worst_n_days} Days", marker="x")
    plt.axhline(worst_pct_threshold, linestyle="--", label=f"Worst {worst_percentile*100:.1f}% Threshold")

    plt.title("Historical Stress Test - Worst Returns Highlighted")
    plt.ylabel("Returns")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==============================
    # 8) Print Key Summary
    # ==============================
    print("========== Stress Testing Module Summary ==========")
    print(f"Portfolio Value: {portfolio_value:,.0f}")
    print(f"Volatility Shock Multiplier: {vol_shock_multiplier}x")
    print("\nTop 10 Worst Daily Returns:")
    print(worst_n)

    print("\nStress Summary Table:")
    print(stress_summary_df.sort_values("Loss (Currency)", ascending=False))

    return stress_summary_df
