import numpy as np
import pandas as pd
from scipy import stats
 
 
def potential_future_exposure(notional, vol, horizon=0.25, confidence=0.95):
    # what could this counterparty owe us in {horizon} years at {confidence} confidence
    # square root of time scaling assumes iid returns — simplification but regulatory standard
    z = stats.norm.ppf(confidence)
    return notional * vol * np.sqrt(horizon) * z
 
 
def expected_positive_exposure(notional, vol, horizon=0.25, n_sims=10000):
    # average exposure across all paths not just the worst
    # more conservative than PFE alone for credit limit setting
    sims = np.random.normal(0, notional * vol * np.sqrt(horizon), n_sims)
    return np.maximum(sims, 0).mean()
 
 
def credit_valuation_adjustment(ead, lgd, pd_annual, maturity=1.0, rf=0.05):
    # CVA = market value of counterparty default risk
    # made mandatory under Basel III post-2008
    # before 2008 most banks didn't price this — cost the industry hundreds of billions
    df = np.exp(-rf * maturity)
    return ead * lgd * pd_annual * df
 
 
def leverage_metrics(gross_exposure, nav, maintenance_margin=0.25):
    gross_lev = gross_exposure / nav
    margin_trigger = nav * maintenance_margin
    distance = nav - margin_trigger
    drawdown_to_trigger = (1 - maintenance_margin / gross_lev)
 
    return {
        'Gross Leverage':             round(gross_lev, 2),
        'Net Asset Value':            round(nav, 0),
        'Margin Trigger ($)':         round(margin_trigger, 0),
        'Distance to Trigger ($)':    round(distance, 0),
        'Max Drawdown Before Margin': f"{drawdown_to_trigger*100:.1f}%"
    }
 
 
def counterparty_scorecard(fund):
    # four dimensions: performance quality, downside risk, leverage, size
    # first-pass credit screening — not a full credit analysis
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
 
    if   score >= 80: rating = 'Investment Grade — Approve'
    elif score >= 55: rating = 'Sub-IG — Approve with Conditions'
    else:             rating = 'High Risk — Decline or Restrict'
 
    return {'Score': score, 'Rating': rating}
 
 
def full_counterparty_report(fund, portfolio_vol, credit_limit=50_000_000):
    notional = fund.get('aum', 100_000_000) * fund.get('leverage', 2)
 
    pfe   = potential_future_exposure(notional, portfolio_vol)
    epe   = expected_positive_exposure(notional, portfolio_vol)
    cva   = credit_valuation_adjustment(
                ead=pfe,
                lgd=0.45,       # standard LGD assumption
                pd_annual=0.02  # 200bps implied default probability
            )
    lev   = leverage_metrics(notional, fund.get('aum', 100_000_000))
    score = counterparty_scorecard(fund)
 
    return {
        'PFE (95%, 3M)':      round(pfe, 0),
        'EPE':                round(epe, 0),
        'CVA':                round(cva, 0),
        'Credit Utilization': f"{min(pfe/credit_limit, 1)*100:.1f}%",
        **lev,
        **score
    }
 