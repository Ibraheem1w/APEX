# APEX

Adaptive Portfolio Exposure and Risk Engine

APEX is a risk project on a portfolio of 7 ETFs. The question behind it is simple. Do the risk numbers for the same portfolio change depending on the kind of market you are in? In this data they do.

## Portfolio

VTV (US value stocks), IWM (US small caps), QUAL (US quality stocks), USMV (US low volatility stocks), TLT (long term Treasuries), TIP (inflation protected Treasuries), and GLD (gold).

The data is daily prices from Yahoo Finance from July 2013 to December 2024, which is 2,882 trading days. It starts in 2013 because that is when QUAL launched.

## What it does

The core folder finds market regimes with a four state hidden Markov model built on SPY returns, volatility, the VIX, and changes in the 10 year yield. The states are named after the model is fit, based on their average return and volatility: Bull, Recovery, Choppy, and High Volatility. The same folder builds a max Sharpe portfolio, with each asset between 5% and 35%, and a risk parity portfolio, both using Ledoit Wolf covariance.

The risk folder covers historical and parametric VaR, CVaR, a Monte Carlo simulation with 10,000 paths over 21 days, crisis correlations, and a correlation network.

The analysis folder has stress tests on 4 real crises and 5 made up scenarios, a Fama French five factor regression with a return breakdown, 8 momentum, volatility, and VIX signals, a Ridge forecast, and a FinBERT sentiment pipeline.

The credit folder works through counterparty exposure, CVA, margin math, and a simple scorecard for a sample leveraged fund.

## Results

The max Sharpe portfolio holds 35% QUAL, 29% GLD, 16% USMV, and 5% in each of the other four. It returned 10.0% a year with 10.9% volatility, a Sharpe ratio of 0.79 using a 1.48% risk free rate, and a worst drawdown of 22.4%.

By regime, the Sharpe ratio was 1.85 in Bull (1,061 days), 1.21 in Recovery (848 days), and 0.29 in Choppy (924 days). The one month VaR at 95% was a 1.7% loss in Bull, 2.9% in Recovery, and 5.6% in Choppy, so the same portfolio carries about three times the risk in a Choppy market. High Volatility caught 18 of the 24 COVID crash days but only has 49 days in total, so it is too short to simulate.

In the stress tests the portfolio lost 22.4% in COVID, 15.0% in the 2022 rate shock, and 5.3% in the 2015 China selloff, and it gained 4.6% during SVB. In the made up scenarios it lost 7.6% if rates rise 3 points, 14.2% if stocks fall 40%, and 19.2% in a liquidity freeze.

Average correlation between the assets was 0.29 over the full period and rose to 0.41 in 2022, when stocks and bonds fell together. It was lower during COVID (0.24) and SVB (close to zero), likely because bonds and gold moved against stocks.

The five factor model explains 79% of daily returns, with a market beta of 0.57. Of the 10.0% yearly return, about 7.2% comes from the market and 1.5% from the risk free rate. Alpha is 1.2% a year but not statistically significant (p = 0.42), and since the factors only cover stocks, some of that is really bond and gold returns.

None of the 8 signals is significant once overlapping 21 day windows are taken into account, and the Ridge forecast does slightly worse than guessing the average (error of 0.0306 against 0.0283).

## Running it

    python -m venv .venv
    source .venv/bin/activate
    pip install numpy pandas scipy scikit-learn yfinance hmmlearn pandas-datareader transformers torch
    python main.py

It needs an internet connection for Yahoo Finance, the Fama French data, and the FinBERT model.

## Limits

Everything is in sample. The weights, regimes, and stress tests all use the same 2013 to 2024 data.

The regime model labels each day using the whole period, so it is not a real time signal. The current regime of Recovery is as of December 30, 2024.

The Monte Carlo assumes normal returns, so it understates how bad the worst days get. Its worst simulated month was a 10.5% loss, while the real worst drawdown was 22.4%.

The made up scenarios, the sample fund, and the CVA inputs (45% loss given default and a 2% default probability) are assumptions, not estimates.

The sentiment step runs on 8 sample headlines. It shows the pipeline works but is not a trading signal.

fast_stats.cpp is a separate C++ standard deviation program and is not used by the Python code.
