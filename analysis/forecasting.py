import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

 
 
FORECAST_HORIZON = 21   # one month ahead
 
def forward_return(pf_ret, horizon=FORECAST_HORIZON):
    return pf_ret.rolling(horizon).sum().shift(-horizon) 
 
def build_signals(returns, vix):
    # momentum: Jegadeesh & Titman (1993)
    # vix: fear gauge as a contrarian predictor
    pf_ret = returns.mean(axis=1)
 
    signals = pd.DataFrame({
        'mom_1m':      pf_ret.rolling(21).mean().shift(1),
        'mom_3m':      pf_ret.rolling(63).mean().shift(1),
        'mom_12m':     pf_ret.rolling(252).mean().shift(1),
 
        # short-term reversal documented in academic lit
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
    # scaler is inside the pipeline so each fold scales on its own training data
    # gap = horizon because the 21-day targets overlap across the split
    pf_ret = returns.mean(axis=1)
    signals = build_signals(returns, vix)
    target = forward_return(pf_ret)
    aligned = signals.join(target.rename('target')).dropna()

    X = aligned.drop('target', axis=1)
    y = aligned['target']

    tscv = TimeSeriesSplit(n_splits=5, gap=FORECAST_HORIZON)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    cv_rmse, base_rmse = [], []
    for tr_idx, te_idx in tscv.split(X):
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = model.predict(X.iloc[te_idx])
        cv_rmse.append(np.sqrt(mean_squared_error(y.iloc[te_idx], preds)))
        base_rmse.append(np.sqrt(np.mean((y.iloc[te_idx] - y.iloc[tr_idx].mean()) ** 2)))

    model.fit(X, y)
    
    return {
        'model':        model,
        'features':     X.columns.tolist(),
        'cv_rmse':      round(np.mean(cv_rmse), 6),
        'base_rmse':    round(np.mean(base_rmse), 6),
        'signal_names': X.columns.tolist()
    }

    
def forecast(trained, recent_returns, vix):
    signals = build_signals(recent_returns, vix)
    last_signal = signals.iloc[[-1]]
    pred = trained['model'].predict(last_signal)[0]
    return round(pred, 5)   # already a 21-day return
 
 
def signal_importance(trained):
    # standardized inputs so coefficients are directly comparable
    coefs = pd.Series(trained['model'][-1].coef_,
                       index=trained['features'])
    return coefs.abs().sort_values(ascending=False)