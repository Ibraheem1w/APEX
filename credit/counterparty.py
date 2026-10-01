import numpy as np
import pandas as pd
from scipy import stats
 
 
def potential_future_exposure(notional, vol, horizon=0.25, confidence=0.95):
    # what could this counterparty owe us in {horizon} years at {confidence} confidence
    # square root of time scaling assumes iid returns, a simplification but a common one
    z = stats.norm.ppf(confidence)
    return notional * vol * np.sqrt(horizon) * z
 
 
def expected_positive_exposure(notional, vol, horizon=0.25, n_sims=10000, seed=42):
    # average positive exposure across simulated paths, not just the worst case
    # lower than PFE, so it suits pricing (CVA) more than setting credit limits
    rng = np.random.default_rng(seed)
    sims = rng.normal(0, notional * vol * np.sqrt(horizon), n_sims)
    return np.maximum(sims, 0).mean()
 

def credit_valuation_adjustment(ead, lgd, pd_annual, maturity=1.0, rf=0.05):
    # CVA = market value of counterparty default risk
    # simplified here: EPE as the exposure, fixed default probability and loss given default
    df = np.exp(-rf * maturity)
    return ead * lgd * pd_annual * df
 
 
def leverage_metrics(gross_exposure, nav, maintenance_margin=0.25):
    # margin call when equity drops below maintenance_margin * position value
    gross_lev = gross_exposure / nav
    loss = (1 / gross_lev - maintenance_margin) / (1 - maintenance_margin)
    loss = max(loss, 0.0)   # 0 means already at or past a margin call

    return {
        'Gross Leverage':      round(gross_lev, 2),
        'Net Asset Value':     round(nav, 0),
        'Required Equity ($)': round(maintenance_margin * gross_exposure, 0),
        'Loss on Exposure Before Margin Call': f"{loss*100:.1f}%"
    }
 
 
def counterparty_scorecard(fund):
    # four dimensions: performance quality, downside risk, leverage, size
    # first-pass credit screening not a full credit analysis
    score = 0
 
    if   fund['sharpe']      >  1.5: score += 25
    elif fund['sharpe']      >  1.0: score += 15
    else:                            score +=  5
 
    if   abs(fund['max_dd']) < 0.10: score += 25
    elif abs(fund['max_dd']) < 0.20: score += 15
    else:                            score +=  5
 
    if   fund['leverage']    <  2.0: score += 25
    elif fund['leverage']    <  4.0: score += 15
    else:                            score +=  5
 
    if   fund['aum']         > 1e9:  score += 25
    elif fund['aum']         > 1e8:  score += 15
    else:                            score +=  5
 
    if   score >= 80: rating = 'Investment Grade: Approve'
    elif score >= 55: rating = 'Sub-IG: Approve with Conditions'
    else:             rating = 'High Risk: Decline or Restrict'
 
    return {'Score': score, 'Rating': rating}
 
 
def full_counterparty_report(fund, portfolio_vol, credit_limit=50_000_000):
    notional = fund.get('aum', 100_000_000) * fund.get('leverage', 2)
 
    pfe   = potential_future_exposure(notional, portfolio_vol)
    epe   = expected_positive_exposure(notional, portfolio_vol)
    cva   = credit_valuation_adjustment(
                ead=epe,
                lgd=0.45,      # 45% is the usual basel number for unsecured, just assuming it
                pd_annual=0.02  # 2% a year, made up, not fitted to anything
            )
    lev   = leverage_metrics(notional, fund.get('aum', 100_000_000))
    score = counterparty_scorecard(fund)
 
    return {
        'PFE (95%, 3M)':      round(pfe, 0),
        'EPE':                round(epe, 0),
        'CVA':                round(cva, 0),
        'Credit Utilization': f"{pfe/credit_limit*100:.1f}%",
        **lev,
        **score
    }
 