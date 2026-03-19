import numpy as np
import pandas as pd
from scipy import stats
from core.portfolio import portfolio_returns, _annualize
 
 
def hist_var(pf_ret, confidence=0.95):
    return np.percentile(pf_ret, (1 - confidence) * 100)
 
 
def cvar(pf_ret, confidence=0.95):
    # VaR just tells you the threshold
    # CVaR tells you the average of what's beyond it — more useful
    # post-2008 most firms shifted to CVaR as the primary metric
    var = hist_var(pf_ret, confidence)
    tail = pf_ret[pf_ret <= var]
    return tail.mean()
 
 
def parametric_var(pf_ret, confidence=0.95):
    # assumes normality — compare against historical as sanity check
    z = stats.norm.ppf(1 - confidence)
    return pf_ret.mean() + z * pf_ret.std()
 
 
def rolling_vol(pf_ret, window=21):
    return pf_ret.rolling(window).std() * np.sqrt(252)
 
 
def rolling_sharpe(pf_ret, window=63, rf=0.05):
    roll_ret = pf_ret.rolling(window).mean() * 252
    roll_vol = pf_ret.rolling(window).std() * np.sqrt(252)
    return (roll_ret - rf) / roll_vol
 
 
def rolling_correlations(returns, window=63):
    # correlations spike during crises
    # diversification collapses exactly when you need it most
    assets = returns.columns
    pairs = {}
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            k = f"{assets[i]}/{assets[j]}"
            pairs[k] = (returns[assets[i]]
                        .rolling(window)
                        .corr(returns[assets[j]]))
    return pd.DataFrame(pairs)
 
 
def crisis_correlation_shift(returns):
    # quantifies how much diversification breaks down in crises
    # most risk models don't surface this explicitly
    crises = {
        'GFC 2008':        ('2008-09-01', '2009-03-31'),
        'Euro Debt 2011':  ('2011-07-01', '2011-10-31'),
        'COVID 2020':      ('2020-02-19', '2020-03-23'),
        'Rate Shock 2022': ('2022-01-01', '2022-10-31'),
        'SVB 2023':        ('2023-03-08', '2023-03-31')
    }
 
    def avg_pairwise(corr_matrix):
        vals = corr_matrix.values
        upper = vals[np.triu_indices_from(vals, k=1)]
        return upper.mean()
 
    baseline = avg_pairwise(returns.corr())
    results = {'Normal Markets': {'Avg Correlation': round(baseline, 4),
                                   'Period': 'Full sample'}}
 
    for label, (start, end) in crises.items():
        try:
            c_ret = returns.loc[start:end]
            if len(c_ret) < 10:
                continue
            c_corr = avg_pairwise(c_ret.corr())
            results[label] = {
                'Avg Correlation':  round(c_corr, 4),
                'Change vs Normal': round(c_corr - baseline, 4),
                'Period':           f"{start} → {end}"
            }
        except Exception:
            continue
 
    return pd.DataFrame(results).T
 
 
def full_risk_report(weights, returns):
    pf = portfolio_returns(weights, returns)
 
    return {
        'VaR 95% (daily)':    round(hist_var(pf, 0.95), 5),
        'VaR 99% (daily)':    round(hist_var(pf, 0.99), 5),
        'CVaR 95% (daily)':   round(cvar(pf, 0.95), 5),
        'Parametric VaR 95%': round(parametric_var(pf, 0.95), 5),
        'VaR/CVaR Ratio':     round(hist_var(pf, 0.95) / cvar(pf, 0.95), 3),
        'Skewness':           round(pf.skew(), 3),
        'Excess Kurtosis':    round(pf.kurt(), 3),
        'Normality (p-val)':  round(stats.normaltest(pf).pvalue, 4)

    }
