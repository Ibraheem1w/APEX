import numpy as np
import pandas as pd
 
# TLT approximate current characteristics
# duration shifts with market — update when rates move significantly
TLT_DURATION = 17.5
TLT_CONVEXITY = 4.2
TLT_TICKER = 'TLT'
TIP_TICKER = 'TIP'
 
 
def macaulay_duration(cash_flows, times, ytm):
    pv = [cf / (1 + ytm)**t for cf, t in zip(cash_flows, times)]
    price = sum(pv)
    dur = sum(t * p / price for t, p in zip(times, pv))
    return dur, price
 
 
def modified_duration(mac_dur, ytm):
    return mac_dur / (1 + ytm)
 
 
def dv01(mod_dur, price, face=1000):
    # bond traders use DV01 not duration — it's in dollar terms
    return mod_dur * price * face / 10000
 
 
def convexity(cash_flows, times, ytm):
    # duration assumes linear price-yield relationship
    # it's actually curved — convexity corrects for that
    # matters a lot for large moves like 2022's +300bps
    pv = [cf / (1 + ytm)**t for cf, t in zip(cash_flows, times)]
    price = sum(pv)
    conv = sum(t * (t+1) * p / ((1 + ytm)**2 * price)
               for t, p in zip(times, pv))
    return conv
 
 
def price_change(mod_dur, conv, dy_bps):
    dy = dy_bps / 10000
    dur_effect  = -mod_dur * dy
    conv_effect = 0.5 * conv * dy**2
    return dur_effect + conv_effect
 
 
def tlt_rate_sensitivity(tlt_weight, portfolio_value,
                          dur=TLT_DURATION, conv=TLT_CONVEXITY):
    # 2022 rate shock was ~300bps so range goes there
    tlt_val = portfolio_value * tlt_weight
    scenarios = [-300, -200, -100, -50, 50, 100, 200, 300]
    rows = []
 
    for bps in scenarios:
        pct = price_change(dur, conv, bps)
        dollar = tlt_val * pct
        rows.append({
            'Rate Move':          f"{bps:+d}bps",
            'TLT Price Change':   f"{pct*100:.2f}%",
            'Portfolio $ Impact': round(dollar, 0)
        })
 
    return pd.DataFrame(rows).set_index('Rate Move')
 
 
def tips_real_rate_analysis(tip_returns, cpi_changes):
    # TIP outperforms when inflation rises faster than nominal yields
    # checking whether that actually holds in the historical data
    aligned = tip_returns.to_frame('TIP').join(
        cpi_changes.rename('CPI')
    ).dropna()
 
    corr = aligned.corr().loc['TIP', 'CPI']
 
    return {
        'TIP/CPI Correlation':     round(corr, 4),
        'Inflation Hedge Quality': 'Strong' if corr > 0.3
                                   else 'Moderate' if corr > 0.1
                                   else 'Weak',
        'Observations':            len(aligned)
    }