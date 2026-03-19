import yfinance as yf
import pandas as pd
import numpy as np
 
# Economic Factor Rotation Portfolio
# Each asset earns its place by serving a specific regime
REGIME_ROTATION_ASSETS = {
    'VTV':  'recovery_leader',
    'IWM':  'growth_amplifier',
    'QUAL': 'downturn_anchor',
    'USMV': 'bear_defensive',
    'TLT':  'recession_hedge',
    'TIP':  'inflation_shield',
    'GLD':  'stagflation_store'
}
 
# Pulled separately — regime detection inputs, not portfolio assets
MACRO_SIGNALS = ['^VIX', 'SPY', '^TNX']
 
 
def fetch_assets(start='2010-01-01', end='2024-12-31'):
    tickers = list(REGIME_ROTATION_ASSETS.keys())
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
    raw = raw.dropna()
 
    pct_returns = raw.pct_change().dropna()
    log_returns = np.log(raw / raw.shift(1)).dropna()
 
    return raw, pct_returns, log_returns
 
 
def fetch_macro(start='2010-01-01', end='2024-12-31'):
    raw = yf.download(MACRO_SIGNALS, start=start, end=end, auto_adjust=True)['Close']
    raw = raw.dropna(how='all').ffill()
 
    vix = raw['^VIX']
    spy_ret = raw['SPY'].pct_change().dropna()
    tny = raw['^TNX']   # 10yr yield
 
    return vix, spy_ret, tny
 
 
def align(prices, macro_series):
    common = prices.index.intersection(macro_series.index)
    return prices.loc[common], macro_series.loc[common]
 
 
# sanity check 
if __name__ == '__main__':
    px, ret, lret = fetch_assets()
    vix, spy, tny = fetch_macro()
    print(f"Assets loaded: {px.shape}")
    print(f"Date range: {px.index[0].date()} → {px.index[-1].date()}")
    print(f"Missing values: {px.isnull().sum().sum()}")