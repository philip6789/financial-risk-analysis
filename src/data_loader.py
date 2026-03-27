
import numpy as np

import yfinance as yf

def load_prices(tickers,start,end):
    raw = yf.download(tickers, start="2018-01-01", end="2023-12-31")
    price = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
    return price

def compute_log_returns(price):
    returns = np.log(price/price.shift(1)).dropna()
    return returns

def build_portfolio_returns(returns,weights):
    portfolio_returns = returns.dot(weights)
    return portfolio_returns

