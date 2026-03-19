import numpy as np
import pandas as pd
 
#  raw returns hide behavioral patterns
# a PM can have positive returns while being systematically bad
# at position sizing or holding losers too long
# these metrics separate skill from luck
 
 
def win_rate(trade_returns):
    winners = (trade_returns > 0).sum()
    return round(winners / len(trade_returns), 4)
 
 
def avg_winner(trade_returns):
    w = trade_returns[trade_returns > 0]
    return round(w.mean(), 5) if len(w) > 0 else 0.0
 
 
def avg_loser(trade_returns):
    l = trade_returns[trade_returns < 0]
    return round(l.mean(), 5) if len(l) > 0 else 0.0
 
 
def profit_factor(trade_returns):
    # total gains / total losses
    # above 1.5 is generally considered disciplined
    # below 1.0 means losses outweigh gains in aggregate
    gross_profit = trade_returns[trade_returns > 0].sum()
    gross_loss   = abs(trade_returns[trade_returns < 0].sum())
    return round(gross_profit / gross_loss, 3) if gross_loss > 0 else np.inf
 
 
def edge_ratio(trade_returns):
    # avg winner / avg loser — tells you if wins are big enough
    # a 40% win rate with edge ratio of 3.0 beats
    # a 60% win rate with edge ratio of 0.8
    aw = avg_winner(trade_returns)
    al = avg_loser(trade_returns)
    return round(abs(aw / al), 3) if al != 0 else np.inf
 
 
def expectancy(trade_returns):
    # expected return per trade — the number that actually matters
    # positive expectancy = viable strategy regardless of win rate
    wr = win_rate(trade_returns)
    aw = avg_winner(trade_returns)
    al = avg_loser(trade_returns)
    return round(wr * aw + (1 - wr) * al, 6)
 
 
def position_sizing_discipline(position_sizes):
    # coefficient of variation on position sizes
    # low CV = consistent sizing = disciplined
    # high CV = erratic sizing = possibly overconfident on certain trades
    sizes = pd.Series(position_sizes)
    return round(sizes.std() / sizes.mean(), 4) if sizes.mean() != 0 else np.nan
 
 
def concentration_score(weights):
    # herfindahl index — standard concentration measure
    # 1.0 = fully concentrated, 1/n = equal weight
    w = np.array(weights)
    return round((w ** 2).sum(), 4)
 
 
def behavior_report(trade_returns, position_sizes=None):
    wr  = win_rate(trade_returns)
    aw  = avg_winner(trade_returns)
    al  = avg_loser(trade_returns)
    pf  = profit_factor(trade_returns)
    er  = edge_ratio(trade_returns)
    exp = expectancy(trade_returns)
 
    report = {
        'Win Rate':             f"{wr*100:.1f}%",
        'Avg Winner':           f"{aw*100:.3f}%",
        'Avg Loser':            f"{al*100:.3f}%",
        'Profit Factor':        pf,
        'Edge Ratio (W/L)':     er,
        'Expectancy per Trade': f"{exp*100:.4f}%"
    }
 
    if position_sizes is not None:
        report['Sizing Discipline (CV)'] = position_sizing_discipline(
            position_sizes
        )
 
    return pd.Series(report, name='PM Behavior Report')