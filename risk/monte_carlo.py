import numpy as np
import pandas as pd
from core.portfolio import portfolio_returns
 
 
def cholesky_simulate(weights, returns, n_sims=10000, horizon=21, seed=42):
   # correlated gaussian draws via cholesky, drift from the sample mean
   # tails are thinner than the real data (excess kurtosis is about 15)
    rng = np.random.default_rng(seed)
    mu = returns.mean().values
    chol = np.linalg.cholesky(returns.cov().values)
    z = rng.standard_normal((n_sims, horizon, len(weights)))
    asset_ret = z @ chol.T + mu
    pf_daily = asset_ret @ np.asarray(weights)
    return (1 + pf_daily).prod(axis=1) - 1
 
 
def simulation_summary(weights, returns, n_sims=10000, horizon=21):
    sims = cholesky_simulate(weights, returns, n_sims, horizon)
 
    return {
        'Expected Return':  round(sims.mean(), 5),
        'VaR 95%':          round(np.percentile(sims, 5), 5),
        'VaR 99%':          round(np.percentile(sims, 1), 5),
        'CVaR 95%':         round(sims[sims <= np.percentile(sims, 5)].mean(), 5),
        'Best Case':        round(sims.max(), 5),
        'Worst Case':       round(sims.min(), 5),
        'Prob of Loss':     round((sims < 0).mean(), 4),
        'Prob Loss > 10%':  round((sims < -0.10).mean(), 4)
    }
 
 
def regime_conditional_simulations(weights, returns, labeled_regimes):
    # aggregate VaR hides regime specific tail risk
    # a portfolio can look fine overall and lose much more in one regime
    results = {}
    for regime in labeled_regimes.unique():
        regime_dates = labeled_regimes[labeled_regimes == regime].index
        r_subset = returns.reindex(regime_dates).dropna()
 
        if len(r_subset) < 63:
            print(f" skipping {regime}: only {len(r_subset)} days") 
            continue
 
        sims = cholesky_simulate(weights, r_subset)
        results[regime] = {
            'VaR 95%':         round(np.percentile(sims, 5), 5),
            'CVaR 95%':        round(sims[sims <= np.percentile(sims, 5)].mean(), 5),
            'Prob of Loss':    round((sims < 0).mean(), 4),
            'Expected Return': round(sims.mean(), 5)
        }
 
    return pd.DataFrame(results).T