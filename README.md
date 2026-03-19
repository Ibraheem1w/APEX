# APEX
## Adaptive Portfolio Exposure and Risk Engine

Most portfolio risk models treat markets as if they're always the same. They're not. A VaR calculated during a bull market is almost meaningless during a crisis because correlations, volatility, and return dynamics are completely different. APEX built around that problem — detect the current regime first, then run everything else conditional on it.

---

## Portfolio

| Ticker | Asset | Role |
|--------|-------|------|
| VTV | Vanguard Value ETF | recovery_leader |
| IWM | iShares Russell 2000 | growth_amplifier |
| QUAL | iShares MSCI Quality | downturn_anchor |
| USMV | iShares Min Volatility | bear_defensive |
| TLT | iShares 20yr Treasury | recession_hedge |
| TIP | iShares TIPS | inflation_shield |
| GLD | SPDR Gold | stagflation_store |

Factor ETFs not individual stocks. Each one is in there because it does something specific in a specific economic environment — not because of views on any particular company.

---

## Modules

**core/regime.py**
4-state Hidden Markov Model on SPY returns, realized vol, and VIX. The model decides which state is which after fitting — you can't hardcode labels before you know what the data looks like. Current regime as of last run: Recovery.

Transition matrix persistence:
- Bull stays Bull 97.4% of days (~38 day average duration)
- Bear stays Bear 97.2% of days (~35 day average)
- High Volatility only persists 88.8% — these regimes are short and violent

**core/optimize.py**
Max Sharpe with Ledoit-Wolf shrinkage vs Risk Parity. With 7 assets and a few hundred observations sample covariance is noisy — small changes in the date range produce very different weights. Ledoit-Wolf fixes that. Risk Parity is included because max Sharpe tends to overweight whatever had the highest recent Sharpe, which isn't always what you want.

**risk/risk_metrics.py**
VaR, CVaR, parametric VaR, rolling correlations, crisis correlation shift. The crisis correlation analysis is the most interesting output — average pairwise correlation in normal markets was 0.2903, jumped to 0.4133 during the 2022 rate shock. That's the diversification breakdown most frameworks miss because they use unconditional full-sample correlations.

**risk/monte_carlo.py**
10,000 path simulation using Cholesky decomposition. Simulating assets independently would lose the correlation structure entirely — Cholesky preserves it. Run both unconditionally and conditional on each regime.

**core/portfolio.py**
Standard metrics plus Information Ratio and regime-conditional stats. The regime breakdown is where it gets interesting:
- Bull: Sharpe 1.626
- Recovery: Sharpe 1.006
- High Volatility: Sharpe -0.245
- Bear: Sharpe -0.030

Unconditional Sharpe of 0.473 hides all of that.

**analysis/factors.py**
Fama-French 5-factor regression with manual OLS — scikit-learn doesn't return standard errors so this is written from scratch. 5-factor not 3-factor because RMW directly captures QUAL's profitability tilt. Using 3-factor would dump that exposure into alpha.

From actual data:
- R-squared: 0.9019
- Annualized alpha: 39.1 bps, t-stat 0.35 — not significant
- Market beta: 0.668
- RMW and CMA both significant at 1% level

**analysis/behavior.py**
Win rate, profit factor, edge ratio, expectancy per trade. Built this because raw returns don't tell you how a strategy is actually working. A 60% win rate with losers twice the size of winners has negative expectancy. A 40% win rate with winners three times the size of losers prints money.

From actual data:
- Win Rate: 54.7%
- Avg Winner: +0.492%, Avg Loser: -0.502%
- Profit Factor: 1.186
- Edge Ratio: 0.98
- Expectancy per Trade: 0.0419%

Edge ratio under 1.0 — the win rate is doing the work here, not the size of winners.

**analysis/attribution.py**
Brinson-Hood-Beebower model plus factor-based decomposition. Two strategies with identical 12% annual returns can get there completely differently — one through genuine selection, one through factor loading that happened to work that year. Attribution separates those cases.

From actual data:
- Stock Selection (Alpha): 0.00391
- Market contribution: 0.03342
- Total Attributed: 0.04592

**analysis/alpha_signals.py**
Information Coefficient across all 8 forecasting signals. IC is rank correlation between predicted and actual returns — more appropriate here than Pearson because returns aren't normally distributed.

Only two signals statistically significant:
- reversal_1w: IC 0.0415, t-stat 2.12
- vix_chg_5d: IC 0.0435, t-stat 2.23

Momentum signals were negative IC. Factor ETFs apparently mean-revert rather than trend — makes sense given how they rebalance.

**analysis/sentiment.py**
FinBERT on financial news headlines. Used FinBERT specifically because financial language is different enough from general text that VADER and TextBlob consistently underperform on things like earnings calls and Fed statements. Output feeds into forecasting as a 9th signal.

**analysis/forecasting.py**
Ridge regression on 8 signals. Ridge not OLS because 1-month and 3-month momentum share variance — OLS inflates them in opposite directions. Walk-forward TimeSeriesSplit, not random splits, because random splits let future data contaminate training.

**analysis/fixed_income.py**
Macaulay duration, modified duration, convexity, DV01 for TLT. Added convexity specifically because 2022 was a +300bps move — at that scale duration alone meaningfully underestimates price changes.

**analysis/stress_testing.py**
Historical scenarios across China Selloff 2015, COVID Crash 2020, Rate Shock 2022, SVB 2023. GFC 2008 and Euro Debt 2011 fall outside the data range. SVB included because it was a credit contagion event not just a market selloff — different risk mechanism than the others.

**credit/counterparty.py**
PFE, EPE, CVA using Basel III framework. CVA became mandatory post-2008 — before that most banks didn't price counterparty default risk at all, which is part of why the losses were so large.

**risk/propagation.py**
Correlation network and minimum spanning tree using Mantegna (1999) distance metric. Pairwise correlations don't tell you how shocks actually travel — the MST shows the load-bearing connections and which assets are structural hubs.

---

## Setup

```bash
pip install yfinance pandas numpy scipy scikit-learn hmmlearn pandas-datareader plotly statsmodels matplotlib transformers torch
```

```bash
python main.py
```

---

## Structure

```
APEX/
├── main.py
├── data.py
├── core/
│   ├── portfolio.py
│   ├── optimize.py
│   └── regime.py
├── risk/
│   ├── risk_metrics.py
│   ├── monte_carlo.py
│   └── propagation.py
├── analysis/
│   ├── stress_testing.py
│   ├── factors.py
│   ├── forecasting.py
│   ├── fixed_income.py
│   ├── alpha_signals.py
│   ├── sentiment.py
│   ├── behavior.py
│   └── attribution.py
└── credit/
    └── counterparty.py
```
