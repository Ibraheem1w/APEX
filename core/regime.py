import numpy as np
import pandas as pd
from hmmlearn import hmm
 
N_REGIMES = 4
 
# labels get assigned after fitting based on avg return of each state
# hardcoding labels before fitting is wrong — the model decides
# which state is which, not us
REGIME_NAMES = {
    0: 'Bull',
    1: 'Recovery',
    2: 'High Volatility',
    3: 'Bear'
}
 
 
def build_features(spy_ret, vix, tny=None):
    vol_21 = spy_ret.rolling(21).std() * np.sqrt(252)
    vol_5  = spy_ret.rolling(5).std()  * np.sqrt(252)
    vol_ratio = (vol_5 / vol_21).replace([np.inf, -np.inf], np.nan)
 
    features = pd.DataFrame({
        'ret':       spy_ret,
        'vol_21':    vol_21,
        'vix':       vix.reindex(spy_ret.index).ffill(),
        'vol_ratio': vol_ratio
    })
 
    if tny is not None:
        features['yield_chg'] = tny.reindex(spy_ret.index).ffill().pct_change()
 
    return features.dropna()
 
 
def fit_hmm(features):
    model = hmm.GaussianHMM(
        n_components=N_REGIMES,
        covariance_type='full',
        n_iter=2000,
        random_state=42,
        tol=1e-5
    )
    model.fit(features.values)
    return model
 
 
def label_states(model, features):
    # sort states by mean return — highest is bull, lowest is bear
    # middle two split by volatility
    states = model.predict(features.values)
    state_series = pd.Series(states, index=features.index)
 
    mean_ret = {}
    mean_vol = {}
    for s in range(N_REGIMES):
        mask = states == s
        mean_ret[s] = features['ret'][mask].mean()
        mean_vol[s] = features['vol_21'][mask].mean()
 
    sorted_by_ret = sorted(mean_ret, key=mean_ret.get, reverse=True)
 
    top2 = sorted_by_ret[:2]
    bot2 = sorted_by_ret[2:]
 
    bull     = max(top2, key=mean_ret.get)
    recovery = min(top2, key=mean_ret.get)
    high_vol = max(bot2, key=mean_vol.get)
    bear     = min(bot2, key=mean_vol.get)
 
    regime_map = {
        bull:     'Bull',
        recovery: 'Recovery',
        high_vol: 'High Volatility',
        bear:     'Bear'
    }
 
    labeled = state_series.map(regime_map)
    return labeled, states, regime_map
 
 
def transition_matrix(model, regime_map):
    # P(Bull → Bull) = 0.97 means ~33 days average before switching
    raw = pd.DataFrame(
        model.transmat_,
        index=range(N_REGIMES),
        columns=range(N_REGIMES)
    )
 
    name_map = {k: v for k, v in regime_map.items()}
    raw = raw.rename(index=name_map, columns=name_map)
    return raw.round(4)
 
 
def regime_persistence(trans_matrix):
    # persistence = 1 / (1 - p_stay)
    result = {}
    for regime in trans_matrix.index:
        p_stay = trans_matrix.loc[regime, regime]
        result[regime] = round(1 / (1 - p_stay), 1)
    return pd.Series(result, name='Avg Days Per Regime')
 
 
def run_regime_analysis(spy_ret, vix, tny=None):
    features = build_features(spy_ret, vix, tny)
    model = fit_hmm(features)
    labeled, states, regime_map = label_states(model, features)
    trans = transition_matrix(model, regime_map)
    persistence = regime_persistence(trans)
 
    return {
        'model':       model,
        'features':    features,
        'labeled':     labeled,
        'states':      states,
        'regime_map':  regime_map,
        'transitions': trans,
        'persistence': persistence,
        'current':     labeled.iloc[-1]
    }