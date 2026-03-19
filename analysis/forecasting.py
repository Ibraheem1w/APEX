import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from core.portfolio import portfolio_returns
 
 
FORECAST_HORIZON = 21   # one month ahead
 
 
def build_signals(returns, vix):
    # momentum: Jegadeesh & Titman (1993)
    # vix: fear gauge as a contrarian predictor
    pf_ret = returns.mean(axis=1)
 
    signals = pd.DataFrame({
        'mom_1m':      pf_ret.rolling(21).mean().shift(1),
        'mom_3m':      pf_ret.rolling(63).mean().shift(1),
        'mom_12m':     pf_ret.rolling(252).mean().shift(1),
 
        # short-term reversal — documented in academic lit
        'reversal_1w': pf_ret.rolling(5).mean().shift(1) * -1,
 
        'vol_21d':     pf_ret.rolling(21).std().shift(1),
        # vol ratio: is current vol elevated vs recent baseline?
        'vol_regime':  (pf_ret.rolling(5).std() /
                        pf_ret.rolling(63).std()).shift(1),
 
        'vix_lvl':     vix.reindex(pf_ret.index).shift(1),
        'vix_chg_5d':  vix.reindex(pf_ret.index).pct_change(5).shift(1)
    }).dropna()
 
    return signals
 
 
def train(returns, vix):
    # Ridge not OLS because 1m and 3m momentum share variance
    # OLS inflates their coefficients in opposite directions
    # Ridge shrinks both toward zero — more stable out of sample
    #
    # TimeSeriesSplit not random because random splits introduce
    # look-ahead bias — you'd be training on the future
    pf_ret = returns.mean(axis=1)
    signals = build_signals(returns, vix)
 
    target = pf_ret.shift(-FORECAST_HORIZON)
    aligned = signals.join(target.rename('target')).dropna()
 
    X = aligned.drop('target', axis=1)
    y = aligned['target']
 
    scaler = StandardScaler()
    X_sc = pd.DataFrame(scaler.fit_transform(X),
                         index=X.index, columns=X.columns)
 
    tscv = TimeSeriesSplit(n_splits=5)
    model = Ridge(alpha=1.0)
 
    cv_rmse = []
    for tr_idx, te_idx in tscv.split(X_sc):
        model.fit(X_sc.iloc[tr_idx], y.iloc[tr_idx])
        preds = model.predict(X_sc.iloc[te_idx])
        rmse = mean_squared_error(y.iloc[te_idx], preds,
                                   squared=False)
        cv_rmse.append(rmse)
 
    model.fit(X_sc, y)
 
    return {
        'model':        model,
        'scaler':       scaler,
        'features':     X.columns.tolist(),
        'cv_rmse':      round(np.mean(cv_rmse), 6),
        'signal_names': X.columns.tolist()
    }
 
 
def forecast(trained, recent_returns, vix):
    signals = build_signals(recent_returns, vix)
    last_signal = signals.iloc[[-1]]
    X_sc = trained['scaler'].transform(last_signal)
    pred = trained['model'].predict(X_sc)[0]
    return round(pred * FORECAST_HORIZON, 5)   # scale to horizon
 
 
def signal_importance(trained):
    # standardized inputs so coefficients are directly comparable
    coefs = pd.Series(trained['model'].coef_,
                       index=trained['features'])
    return coefs.abs().sort_values(ascending=False)