import numpy as np
import pandas as pd
 
TRADING_DAYS = 252
RF_RATE = 0.05   # update when environment changes
 
 
def _annualize(daily_val, square_root=False):
    # tired of writing * 252 or * sqrt(252) everywhere
    return daily_val * np.sqrt(TRADING_DAYS) if square_root else daily_val * TRADING_DAYS
 
 
def portfolio_returns(weights, returns):
    w = np.array(weights)
    return returns.dot(w)
 
 
def annualized_return(weights, returns):
    pf = portfolio_returns(weights, returns)
    return _annualize(pf.mean())
 
 
def annualized_vol(weights, cov_matrix):
    w = np.array(weights)
    daily_var = w.T @ cov_matrix @ w
    return _annualize(np.sqrt(daily_var), square_root=True)
 
 
def sharpe(weights, returns, cov_matrix, rf=RF_RATE):
    ret = annualized_return(weights, returns)
    vol = annualized_vol(weights, cov_matrix)
    return (ret - rf) / vol
 
 
def information_ratio(pf_returns, benchmark_returns):
    # active return / tracking error
    # Sharpe ignores the benchmark — IR doesn't
    active = pf_returns - benchmark_returns
    te = _annualize(active.std(), square_root=True)
    active_ann = _annualize(active.mean())
    return active_ann / te
 
 
def max_drawdown(pf_returns):
    cumulative = (1 + pf_returns).cumprod()
    rolling_peak = cumulative.cummax()
    dd = (cumulative - rolling_peak) / rolling_peak
    return dd.min()
 
 
def calmar(weights, returns):
    pf = portfolio_returns(weights, returns)
    ann_ret = _annualize(pf.mean())
    mdd = max_drawdown(pf)
    return ann_ret / abs(mdd)
 
 
def regime_conditional_stats(weights, returns, states, regime_map):
    # unconditional Sharpe hides regime-specific bleed
    # a strategy can look fine overall and lose badly in bear markets
    pf = portfolio_returns(weights, returns)
    out = {}
 
    for state_id, label in regime_map.items():
        mask = states == state_id
        r = pf[mask]
 
        if len(r) < 21:
            continue
 
        vol = _annualize(r.std(), square_root=True)
        ann_ret = _annualize(r.mean())
 
        out[label] = {
            'Ann Return':   round(ann_ret, 4),
            'Volatility':   round(vol, 4),
            'Sharpe':       round((ann_ret - RF_RATE) / vol, 3) if vol > 0 else np.nan,
            'Max Drawdown': round(max_drawdown(r), 4),
            'Obs':          int(mask.sum())
        }
 
    return pd.DataFrame(out).T
 
 
def full_scorecard(weights, returns, cov_matrix, benchmark_returns=None):
    pf = portfolio_returns(weights, returns)
    ann_ret = _annualize(pf.mean())
    vol = annualized_vol(weights, cov_matrix)
 
    result = {
        'Annualized Return': round(ann_ret, 4),
        'Annualized Vol':    round(vol, 4),
        'Sharpe Ratio':      round(sharpe(weights, returns, cov_matrix), 3),
        'Max Drawdown':      round(max_drawdown(pf), 4),
        'Calmar Ratio':      round(calmar(weights, returns), 3),
        'Skewness':          round(pf.skew(), 3),
        'Excess Kurtosis':   round(pf.kurt(), 3)
    }
 
    if benchmark_returns is not None:
        bm_aligned = benchmark_returns.reindex(pf.index).dropna()
        pf_aligned = pf.reindex(bm_aligned.index)
        result['Information Ratio'] = round(
            information_ratio(pf_aligned, bm_aligned), 3
        )
 
    return pd.Series(result)