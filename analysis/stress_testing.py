import numpy as np
import pandas as pd
from core.portfolio import portfolio_returns
 
HISTORICAL_CRISES = {
    'GFC 2008':          ('2008-09-01', '2009-03-31'),
    'Euro Debt 2011':    ('2011-07-01', '2011-10-31'),
    'China Selloff 2015':('2015-06-01', '2015-09-30'),
    'COVID Crash 2020':  ('2020-02-19', '2020-03-23'),
    'Rate Shock 2022':   ('2022-01-01', '2022-10-31'),
    'SVB Crisis 2023':   ('2023-03-08', '2023-03-31')
}
 
# SVB included because it was a credit contagion event not just
# a market selloff — different risk mechanism than the others
 
HYPOTHETICAL_SHOCKS = {
    'Rates +300bps': {
        'VTV': -0.08, 'IWM': -0.10, 'QUAL': -0.06,
        'USMV': -0.06, 'TLT': -0.28, 'TIP': -0.10, 'GLD': -0.05
    },
    'Equity -40%': {
        'VTV': -0.40, 'IWM': -0.45, 'QUAL': -0.30,
        'USMV': -0.25, 'TLT':  0.15, 'TIP':  0.05, 'GLD':  0.12
    },
    'Stagflation': {
        'VTV': -0.10, 'IWM': -0.15, 'QUAL': -0.05,
        'USMV': -0.05, 'TLT': -0.20, 'TIP':  0.12, 'GLD':  0.25
    },
    'Liquidity Freeze': {
        'VTV': -0.32, 'IWM': -0.42, 'QUAL': -0.28,
        'USMV': -0.22, 'TLT':  0.05, 'TIP': -0.03, 'GLD': -0.08
    },
    'Soft Landing': {
        'VTV':  0.12, 'IWM':  0.18, 'QUAL':  0.08,
        'USMV':  0.05, 'TLT':  0.06, 'TIP':  0.03, 'GLD': -0.03
    }
}
 
 
def historical_stress(prices, weights, crises=None):
    if crises is None:
        crises = HISTORICAL_CRISES
 
    results = {}
    for name, (start, end) in crises.items():
        try:
            period = prices.loc[start:end]
            if len(period) < 5:
                continue
            ret = period.pct_change().dropna()
            pf = portfolio_returns(weights, ret)
            results[name] = {
                'Total Return': round(pf.sum(), 4),
                'Worst Day':    round(pf.min(), 4),
                'Realized Vol': round(pf.std() * np.sqrt(252), 4),
                'Days':         len(ret)
            }
        except Exception:
            continue
 
    return pd.DataFrame(results).T
 
 
def hypothetical_stress(weights, asset_names, shocks=None):
    if shocks is None:
        shocks = HYPOTHETICAL_SHOCKS
 
    results = {}
    for scenario, asset_shocks in shocks.items():
        total = 0.0
        for asset, shock in asset_shocks.items():
            if asset in asset_names:
                idx = list(asset_names).index(asset)
                total += weights[idx] * shock
        results[scenario] = round(total, 4)
 
    return pd.Series(results, name='Portfolio Impact')
 
 
def worst_case_summary(prices, weights, asset_names):
    hist = historical_stress(prices, weights)
    hypo = hypothetical_stress(weights, asset_names)
 
    print("historical stress results:")
    print(hist[['Total Return', 'Worst Day']].to_string())
    print("\nhypothetical scenarios:")
    print(hypo.to_string())
 
    return hist, hypo