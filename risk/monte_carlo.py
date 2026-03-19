import numpy as np
import pandas as pd
from core.portfolio import portfolio_returns
 
 
def cholesky_simulate(weights, returns, n_sims=10000, horizon=21):
    # simulating assets independently loses correlation structure
    # cholesky preserves it — matters a lot for tail risk estimates
    cov = returns.cov().values
    chol = np.linalg.cholesky(cov)
    n_assets = len(weights)
    w = np.array(weights)
 
    sim_outcomes = np.empty(n_sims)
 
    for i in range(n_sims):
        shocks = np.random.standard_normal((n_assets, horizon))
        corr_shocks = chol @ shocks        # shape: (n_assets, horizon)
        asset_ret = corr_shocks.T          # (horizon, n_assets)
        pf_daily = asset_ret @ w
        sim_outcomes[i] = (1 + pf_daily).prod() - 1
 
    return sim_outcomes
 
 
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
    # aggregate VaR hides regime-specific tail risk
    # a portfolio can look fine overall and bleed badly in bear markets
    results = {}
    for regime in labeled_regimes.unique():
        regime_dates = labeled_regimes[labeled_regimes == regime].index
        r_subset = returns.reindex(regime_dates).dropna()
 
        if len(r_subset) < 63:
            continue
 
        sims = cholesky_simulate(weights, r_subset)
        results[regime] = {
            'VaR 95%':         round(np.percentile(sims, 5), 5),
            'CVaR 95%':        round(sims[sims <= np.percentile(sims, 5)].mean(), 5),
            'Prob of Loss':    round((sims < 0).mean(), 4),
            'Expected Return': round(sims.mean(), 5)
        }
 
    return pd.DataFrame(results).T