# APEX — Adaptive Portfolio Exposure and Risk Engine
#
# core idea: unconditional risk metrics lie.
# a portfolio's VaR in a bull market tells you nothing
# about what happens when the regime shifts.
# built this to see if conditioning on regime state
# actually changes the numbers meaningfully. it does.

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from data import fetch_assets, fetch_macro
from core.portfolio import full_scorecard, regime_conditional_stats, max_drawdown
from core.optimize import max_sharpe, risk_parity, compare_allocations, robust_cov
from core.regime import run_regime_analysis
from risk.risk_metrics import full_risk_report, crisis_correlation_shift
from risk.monte_carlo import simulation_summary, regime_conditional_simulations
from risk.propagation import full_shock_matrix, minimum_spanning_tree
from analysis.stress_testing import historical_stress, hypothetical_stress
from analysis.factors import fetch_ff5, run_factor_model, alpha_decomposition
from analysis.forecasting import train as train_forecast, forecast, signal_importance
from analysis.alpha_signals import evaluate_all_signals, ic_summary
from analysis.sentiment import sentiment_signal
from analysis.behavior import behavior_report
from analysis.attribution import simple_attribution
from analysis.fixed_income import tlt_rate_sensitivity
from credit.counterparty import full_counterparty_report


def run():

    print("\nfetching prices...")
    prices, returns, log_ret = fetch_assets()
    vix, spy_ret, tny = fetch_macro()

    # vix and returns don't always have identical trading day
    # coverage so aligning on common index before anything else
    common_idx = returns.index.intersection(vix.index)
    returns_aligned = returns.loc[common_idx]
    vix_aligned = vix.loc[common_idx]

    print("\ndetecting regimes...")
    regime_out = run_regime_analysis(spy_ret, vix_aligned, tny)
    print(f"current regime: {regime_out['current']}")
    print("\ntransition matrix:")
    print(regime_out['transitions'].to_string())
    print("\nregime persistence (avg days):")
    print(regime_out['persistence'].to_string())

    print("\noptimizing weights...")
    cov = robust_cov(returns_aligned)
    ms_weights = max_sharpe(returns_aligned).values
    rp_weights = risk_parity(returns_aligned).values

    print("\nmax sharpe vs risk parity:")
    print(compare_allocations(returns_aligned).to_string())

    print("\nportfolio scorecard:")
    scorecard = full_scorecard(ms_weights, returns_aligned, cov.values)
    print(scorecard.to_string())

    print("\nperformance by regime:")
    regime_stats = regime_conditional_stats(
        ms_weights, returns_aligned,
        regime_out['states'],
        regime_out['regime_map']
    )
    print(regime_stats.to_string())

    print("\nrisk metrics...")
    risk = full_risk_report(ms_weights, returns_aligned)
    for k, v in risk.items():
        print(f"  {k}: {v}")

    print("\ncorrelation shift during crises:")
    print(crisis_correlation_shift(returns_aligned).to_string())

    print("\nmonte carlo (10k paths, 21-day horizon)...")
    mc = simulation_summary(ms_weights, returns_aligned)
    for k, v in mc.items():
        print(f"  {k}: {v}")

    print("\nstress testing...")
    hist_stress = historical_stress(prices, ms_weights)
    hypo_stress = hypothetical_stress(ms_weights, returns_aligned.columns)
    print(hist_stress.to_string())
    print(hypo_stress.to_string())

    print("\nstress transmission network...")
    shock_matrix = full_shock_matrix(returns_aligned)
    mst = minimum_spanning_tree(returns_aligned)
    print(shock_matrix.round(4).to_string())
    print("\nrisk backbone (MST):")
    print(mst.to_string())

    # store ff_results at outer scope so attribution can use it
    ff_results = None

    print("\nfactor attribution...")
    try:
        start_str = returns_aligned.index[0].strftime('%Y-%m-%d')
        end_str   = returns_aligned.index[-1].strftime('%Y-%m-%d')
        ff5 = fetch_ff5(start_str, end_str)
        pf_ret = returns_aligned.dot(ms_weights)
        ff_results, r2, adj_r2 = run_factor_model(pf_ret, ff5)
        alpha = alpha_decomposition(pf_ret, ff5)
        print(ff_results[['Coefficient', 'T-Statistic',
                           'P-Value', 'Significant']].to_string())
        print(f"\nr-squared: {r2:.4f}")
        print(f"annualized alpha: {alpha['Annualized Alpha (bps)']:.1f} bps")
        print(f"alpha significant: {alpha['Alpha Significant']}")
    except Exception as e:
        print(f"factor data unavailable: {e}")

    print("\nPM behavior analytics...")
    try:
        pf_ret_behavior = returns_aligned.dot(ms_weights)
        report = behavior_report(pf_ret_behavior)
        print(report.to_string())
    except Exception as e:
        print(f"behavior analytics unavailable: {e}")

    print("\nreturn attribution...")
    try:
        if ff_results is not None:
            attr = simple_attribution(ms_weights, returns_aligned, ff_results)
            print(attr.to_string())
        else:
            print("attribution unavailable: factor model unavailable")
    except Exception as e:
        print(f"attribution unavailable: {e}")

    print("\nalpha signal evaluation...")
    try:
        ic_results = evaluate_all_signals(returns_aligned, vix_aligned)
        print(ic_results[['IC', 'IC T-Stat', 'Significant']].to_string())
    except Exception as e:
        print(f"alpha signal evaluation unavailable: {e}")

    print("\nsentiment analysis...")
    try:
        sentiment_summary, sentiment_scores = sentiment_signal()
        for k, v in sentiment_summary.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"sentiment unavailable: {e}")

    print("\ncounterparty credit assessment...")
    pf_ret = returns_aligned.dot(ms_weights)
    sample_fund = {
        'sharpe':   round((pf_ret.mean()*252 - 0.05) /
                          (pf_ret.std()*np.sqrt(252)), 2),
        'max_dd':   max_drawdown(pf_ret),
        'leverage': 2.5,
        'aum':      500_000_000
    }
    credit_report = full_counterparty_report(
        sample_fund,
        portfolio_vol=pf_ret.std() * np.sqrt(252)
    )
    for k, v in credit_report.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    run()
