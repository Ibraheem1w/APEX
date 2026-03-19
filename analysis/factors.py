import numpy as np
import pandas as pd
import pandas_datareader as pdr
from scipy import stats
 
 
FF5_FACTORS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
 
FACTOR_DESCRIPTIONS = {
    'Mkt-RF': 'Market risk premium',
    'SMB':    'Small minus big (size)',
    'HML':    'High minus low (value)',
    'RMW':    'Robust minus weak (profitability)',
    'CMA':    'Conservative minus aggressive (investment)'
}
 
# using 5-factor not 3-factor because RMW captures QUAL's
# profitability exposure directly. 3-factor would misattribute
# that as alpha.
 
 
def fetch_ff5(start, end):
    ff = pdr.get_data_famafrench(
        'F-F_Research_Data_5_Factors_2x3_daily',
        start=start, end=end
    )[0] / 100
    ff.index = pd.to_datetime(ff.index, format='%Y%m%d')
    return ff
 
 
def ols_with_stats(y, X):
    # manual OLS because scikit-learn doesn't return standard errors
    X_c = np.column_stack([np.ones(len(X)), X])
    n, k = X_c.shape
 
    beta = np.linalg.lstsq(X_c, y, rcond=None)[0]
    y_hat = X_c @ beta
    residuals = y - y_hat
 
    sse = residuals @ residuals
    sigma2 = sse / (n - k)
    var_beta = sigma2 * np.linalg.inv(X_c.T @ X_c)
    se = np.sqrt(np.diag(var_beta))
    t_stats = beta / se
    p_vals = [2 * (1 - stats.t.cdf(abs(t), df=n-k)) for t in t_stats]
 
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - sse / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
 
    return beta, se, t_stats, p_vals, r2, adj_r2
 
 
def run_factor_model(portfolio_returns, ff5):
    aligned = portfolio_returns.to_frame('port').join(ff5).dropna()
 
    y = (aligned['port'] - aligned['RF']).values
    X = aligned[FF5_FACTORS].values
 
    beta, se, t_stats, p_vals, r2, adj_r2 = ols_with_stats(y, X)
 
    labels = ['Alpha'] + FF5_FACTORS
    results = pd.DataFrame({
        'Coefficient': beta,
        'Std Error':   se,
        'T-Statistic': t_stats,
        'P-Value':     p_vals,
        'Significant': [p < 0.05 for p in p_vals],
        'Description': ['Unexplained alpha'] + [FACTOR_DESCRIPTIONS[f]
                                                  for f in FF5_FACTORS]
    }, index=labels)
 
    return results, r2, adj_r2
 
 
def rolling_betas(portfolio_returns, ff5, window=126):
    # drifting beta means the strategy is changing character
    # stable beta means it's doing what it's supposed to do
    aligned = portfolio_returns.to_frame('port').join(ff5).dropna()
    excess = aligned['port'] - aligned['RF']
 
    rolling_beta = (excess
                    .rolling(window)
                    .cov(aligned['Mkt-RF'])
                    / aligned['Mkt-RF'].rolling(window).var())
 
    return rolling_beta.dropna()
 
 
def alpha_decomposition(portfolio_returns, ff5):
    results, r2, adj_r2 = run_factor_model(portfolio_returns, ff5)
 
    alpha_bps = results.loc['Alpha', 'Coefficient'] * 252 * 10000
    alpha_sig = results.loc['Alpha', 'Significant']
 
    return {
        'Annualized Alpha (bps)': round(alpha_bps, 1),
        'Alpha T-Stat':           round(results.loc['Alpha', 'T-Statistic'], 3),
        'Alpha Significant':      alpha_sig,
        'R-Squared':              round(r2, 4),
        'Adj R-Squared':          round(adj_r2, 4),
        'Factor-Explained (%)':   round(r2 * 100, 1)
    }