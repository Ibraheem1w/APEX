import numpy as np
import pandas as pd
 
# two PMs can have identical returns via completely different routes
# one through genuine stock selection, one through factor exposure
# that happened to do well that year
# attribution separates those two cases
 
 
def factor_attribution(portfolio_returns, factor_returns,
                        factor_exposures):
    # decompose returns into factor contributions
    # residual = what's left after factors = selection effect
    # this is the number PCAT cares most about
    factor_contrib = factor_exposures * factor_returns
    total_factor   = factor_contrib.sum()
    residual       = portfolio_returns - total_factor
 
    result = factor_contrib.to_dict()
    result['Selection (Residual)'] = round(residual, 5)
    result['Total Return']         = round(portfolio_returns, 5)
    result['Factor Explained (%)'] = round(
        total_factor / portfolio_returns * 100, 1
    ) if portfolio_returns != 0 else np.nan
 
    return pd.Series(result)
 
 
def brinson_attribution(portfolio_weights, benchmark_weights,
                         portfolio_returns, benchmark_returns,
                         sector_map):
    # brinson-hood-beebower model
    # industry standard for decomposing active returns into:
    # allocation effect — sector over/underweights
    # selection effect — stock picking within sectors
    # interaction — combination of both
    sectors = sector_map.unique()
    results = []
 
    for sector in sectors:
        assets = sector_map[sector_map == sector].index
        assets = [a for a in assets if a in portfolio_weights.index]
 
        if not assets:
            continue
 
        pw  = portfolio_weights[assets].sum()
        bw  = benchmark_weights.reindex(assets).fillna(0).sum()
        pr  = (portfolio_returns[assets] *
               portfolio_weights[assets]).sum() / pw if pw > 0 else 0
        br  = (benchmark_returns.reindex(assets).fillna(0) *
               benchmark_weights.reindex(assets).fillna(0)).sum() / bw \
               if bw > 0 else 0
        total_br = (benchmark_returns * benchmark_weights).sum()
 
        allocation  = (pw - bw) * (br - total_br)
        selection   = bw * (pr - br)
        interaction = (pw - bw) * (pr - br)
 
        results.append({
            'Sector':       sector,
            'Port Weight':  round(pw, 4),
            'Bench Weight': round(bw, 4),
            'Allocation':   round(allocation, 5),
            'Selection':    round(selection, 5),
            'Interaction':  round(interaction, 5),
            'Total Active': round(allocation + selection + interaction, 5)
        })
 
    return pd.DataFrame(results).set_index('Sector')
 
 
def rolling_attribution(weights, returns, factor_returns,
                         factor_exposures, window=63):
    # tracks how attribution shifts over time
    # a PM whose alpha is shrinking while factor exposure grows
    # is losing their edge — this surfaces that
    pf_returns = returns.dot(weights)
    results = []
 
    for i in range(window, len(pf_returns)):
        period_ret    = pf_returns.iloc[i-window:i].sum()
        period_factor = (factor_exposures *
                         factor_returns.iloc[i-window:i].sum())
        residual      = period_ret - period_factor.sum()
 
        results.append({
            'date':    pf_returns.index[i],
            'alpha':   round(residual, 5),
            'factor':  round(period_factor.sum(), 5),
            'total':   round(period_ret, 5)
        })
 
    return pd.DataFrame(results).set_index('date')
 
 
def simple_attribution(weights, returns, factor_model_results):
    # quick version using existing FF5 output
    # reframes what the factor regression already calculated
    # no need to rebuild from scratch
    total_return = returns.dot(weights).sum() * 252
    contributions = {}
 
    for factor, row in factor_model_results.iterrows():
        if factor == 'Alpha':
            contributions['Stock Selection (Alpha)'] = round(
                row['Coefficient'] * 252, 5
            )
        else:
            contributions[factor] = round(row['Coefficient'] * 0.05, 5)
 
    contributions['Total Attributed'] = sum(contributions.values())
    return pd.Series(contributions)