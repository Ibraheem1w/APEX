import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from core.portfolio import sharpe, annualized_vol, TRADING_DAYS
 
 
def robust_cov(returns):
    # sample covariance breaks down with limited data relative to assets
    # ledoit-wolf shrinks extreme off-diagonal entries toward a structured target
    lw = LedoitWolf()
    lw.fit(returns)
    return pd.DataFrame(lw.covariance_,
                        index=returns.columns,
                        columns=returns.columns)
 
 
def max_sharpe(returns, max_weight=0.35, min_weight=0.05):
    cov = robust_cov(returns)
    n = len(returns.columns)
    w0 = np.array([1/n] * n)
 
    def neg_sr(w):
        return -sharpe(w, returns, cov.values)
 
    bounds = [(min_weight, max_weight)] * n
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
 
    res = minimize(neg_sr, w0, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-9, 'maxiter': 1000})
 
    return pd.Series(res.x.round(4), index=returns.columns)
 
 
def risk_parity(returns):
    # each asset contributes equally to total portfolio variance
    # mean-variance over-concentrates in high-vol assets
    # risk parity sidesteps that — conceptually close to Bridgewater All Weather
    cov = robust_cov(returns).values
    n = len(returns.columns)
    w0 = np.array([1/n] * n)
 
    def rp_objective(w):
        pf_vol = np.sqrt(w @ cov @ w)
        marginal = cov @ w / pf_vol
        contrib = w * marginal
        target = pf_vol / n
        return np.sum((contrib - target) ** 2)
 
    bounds = [(0.01, 0.6)] * n
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
 
    res = minimize(rp_objective, w0, method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-10, 'maxiter': 2000})
 
    return pd.Series(res.x.round(4), index=returns.columns)
 
 
def efficient_frontier(returns, n_points=2000):
    cov = robust_cov(returns)
    n = len(returns.columns)
    results = []
 
    for _ in range(n_points):
        w = np.random.dirichlet(np.ones(n))
        ret = (returns.mean() * w).sum() * TRADING_DAYS
        vol = annualized_vol(w, cov.values)
        sr = sharpe(w, returns, cov.values)
        results.append({'Return': ret, 'Volatility': vol,
                        'Sharpe': sr, 'Weights': w})
 
    return pd.DataFrame(results)
 
 
def compare_allocations(returns):
    # the disagreements between methods are often more interesting than the outputs
    ms = max_sharpe(returns)
    rp = risk_parity(returns)
 
    return pd.DataFrame({
        'Max Sharpe':  ms,
        'Risk Parity': rp,
        'Difference':  (ms - rp).round(4)
    })