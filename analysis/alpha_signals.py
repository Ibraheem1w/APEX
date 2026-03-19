import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from analysis.forecasting import build_signals, FORECAST_HORIZON
 
 
# IC > 0.05 is generally considered useful in practice
# most academic signals decay within 5-10 days
# rolling IC tells you whether the signal is stable or losing its edge
 
 
def information_coefficient(predicted, actual):
    # rank correlation between predicted and actual returns
    # better than pearson here because returns aren't normally distributed
    ic, _ = spearmanr(predicted, actual)
    return ic
 
 
def ic_rolling(predictions, actuals, window=63):
    # rolling IC shows whether signal quality is stable over time
    # a decaying IC means the signal is losing predictive power
    ics = []
    for i in range(window, len(predictions)):
        ic = information_coefficient(
            predictions[i-window:i],
            actuals[i-window:i]
        )
        ics.append(ic)
    return pd.Series(ics, name='Rolling IC')
 
 
def signal_decay(predictions, actuals, horizons=None):
    # how far into the future does the signal have predictive power
    # most signals peak at 1-5 days then decay rapidly
    if horizons is None:
        horizons = [1, 5, 10, 21]
 
    results = {}
    for h in horizons:
        if h >= len(predictions):
            continue
        ic = information_coefficient(
            predictions[:-h],
            actuals[h:]
        )
        results[f'{h}d'] = round(ic, 4)
 
    return pd.Series(results, name='Signal Decay by Horizon')
 
 
def ic_summary(predictions, actuals):
    # full evaluation of a signal — IC, t-stat, and decay
    ic = information_coefficient(predictions, actuals)
 
    # t-stat for IC significance
    n = len(predictions)
    t_stat = ic * np.sqrt(n) / np.sqrt(1 - ic**2) if abs(ic) < 1 else np.nan
 
    decay = signal_decay(predictions, actuals)
 
    return {
        'IC':            round(ic, 4),
        'IC T-Stat':     round(t_stat, 3),
        'Significant':   abs(t_stat) > 1.96 if not np.isnan(t_stat) else False,
        'IC Annualized': round(ic * np.sqrt(252), 4),
        'Signal Decay':  decay.to_dict()
    }
 
 
def evaluate_all_signals(returns, vix):
    # run IC evaluation across all 8 signals in the forecasting module
    # shows which signals actually have predictive power
    signals = build_signals(returns, vix)
    pf_ret = returns.mean(axis=1)
    target = pf_ret.shift(-FORECAST_HORIZON).reindex(signals.index).dropna()
    signals_aligned = signals.reindex(target.index)
 
    results = {}
    for signal_name in signals_aligned.columns:
        sig = signals_aligned[signal_name].dropna()
        tgt = target.reindex(sig.index)
        common = sig.dropna().index.intersection(tgt.dropna().index)
 
        if len(common) < 100:
            continue
 
        results[signal_name] = ic_summary(
            sig.loc[common].values,
            tgt.loc[common].values
        )
 
    return pd.DataFrame(results).T