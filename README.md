# financial-risk-analysis
A financial risk analysis project covering VaR estimation, backtesting (Kupiec &amp; Christoffersen), and volatility modeling using EWMA and GARCH.


## 📌 Overview

This project builds a comprehensive financial risk analysis framework for a multi-asset portfolio (AAPL, MSFT, GOOGL).

It covers the full pipeline from data processing to risk modeling, backtesting, and stress testing.

---

## ⚙️ Methodology

### 1. Data Processing
- Download historical price data
- Compute log returns
- Construct weighted portfolio returns

---

### 2. Value-at-Risk (VaR) Models
- Historical VaR (non-parametric)
- Parametric VaR (normal assumption)
- Rolling window estimation (250-day)

---

### 3. Backtesting Framework
- Exception detection (VaR violations)
- Kupiec test (unconditional coverage)
- Christoffersen test (independence)

---

### 4. Volatility Modeling
- EWMA volatility model (λ = 0.94)
- Rolling GARCH(1,1) with periodic refitting

---

### 5. Stress Testing & Scenario Analysis
- Worst-case historical losses
- Tail risk (percentile-based)
- Hypothetical shocks (-10%, -20%, -30%)
- Volatility shock scenarios

---

## 📊 Key Features

- Modular design with reusable risk models (`src/`)
- Rolling estimation capturing time-varying risk
- Statistical validation of risk models
- Comparison across multiple VaR approaches
- Integrated stress testing framework

---

## 📂 Project Structure
\---financial analysis-github
    +---notebooks
    |       main.ipynb
    |
    \---src
            backtesting.py
            bug_test.py
            compute_var.py
            data_loader.py
            rolling_var.py
            stress_testing.py
            volatility.py

## ⚙️ Environment Setup

```bash
pip install -r requirements.txt

## Tested with
Python 3.10
