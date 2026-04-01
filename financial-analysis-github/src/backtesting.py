
import pandas as pd
import matplotlib.pyplot  as plt
from scipy.stats import chi2
import numpy as np


def compute_exceptions(returns, var_series):
    exceptions = []

    for r, v in zip(returns, var_series):
        if r < v:
            exceptions.append(1)
        else:
            exceptions.append(0)

    return pd.Series(exceptions, index=returns.index)

def exception_rate(exceptions):
    x = exceptions.sum()
    t = len(exceptions)
    p_hat = x / t

    return p_hat, x, t



def kupiec_test(x, t, alpha):
    if x == 0 or x == t:
        return np.nan, np.nan

    p_hat = x / t
    lr_uc = -2 * np.log(
        ((1 - alpha)**(t - x) * (alpha**x)) /
        ((1 - p_hat)**(t - x) * (p_hat**x))
    )

    p_value = 1 - chi2.cdf(lr_uc, df=1)

    if p_value < 0.05:
        decision = "Reject"
    else:
        decision = "Pass"

    return lr_uc, p_value, #decision :allow flexible significance levels and consistent model governance.

def plot_exceptions(returns, var_series, exceptions):

    plt.figure(figsize=(12,6))
    plt.plot(returns, label="Returns")
    plt.plot(-var_series, label="VaR", color="black")

    exception_points = returns[exceptions == 1]
    plt.scatter(exception_points.index,
                exception_points.values,
                marker="o",
                label="Exceptions")

    plt.legend()
    plt.title("VaR Backtesting: Exceptions Plot")
    plt.show()






def christoffersen_test(exceptions):

    n00 = n01 = n10 = n11 = 0

    for t in range(1, len(exceptions)):
        if exceptions.iloc[t - 1] == 0 and exceptions.iloc[t] == 0:
            n00 += 1
        elif exceptions.iloc[t - 1] == 0 and exceptions.iloc[t] == 1:
            n01 += 1
        elif exceptions.iloc[t - 1] == 1 and exceptions.iloc[t] == 0:
            n10 += 1
        elif exceptions.iloc[t - 1] == 1 and exceptions.iloc[t] == 1:
            n11 += 1

    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)

    l0 = (1 - p) ** (n00 + n10) * p ** (n01 + n11)
    l1 = ((1 - p01) ** n00) * (p01 ** n01) * ((1 - p11) ** n10) * (p11 ** n11)

    lr_ind = -2 * np.log(l0 / l1)
    p_value = 1 - chi2.cdf(lr_ind, df=1)

    return lr_ind, p_value

def plot_rolling_var(returns,rolling_var,exceptions,title="Rolling VaR Backtesting"):

    plt.figure(figsize=(12, 6))
    plt.plot(returns.index, returns.values, label="Returns")
    plt.plot(rolling_var.index, rolling_var.values, label="Rolling VaR", color="red")

    exc_idx = exceptions[exceptions == 1].index
    plt.scatter(exc_idx, returns.loc[exc_idx], color="black", label="Exceptions")

    plt.legend()
    plt.title(title)
    plt.show()



