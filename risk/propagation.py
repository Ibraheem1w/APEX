import numpy as np
import pandas as pd
 
# TODO: add networkx visualization for the dashboard
# keeping computation separate from rendering for now
 
 
def correlation_network(returns, threshold=0.3):
    # assets as nodes, strong correlations as edges
    # shows which assets move together during stress
    corr = returns.corr()
    assets = corr.columns.tolist()
    edges = []
 
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            c = corr.iloc[i, j]
            if abs(c) >= threshold:
                edges.append({
                    'source': assets[i],
                    'target': assets[j],
                    'weight': round(c, 4)
                })
 
    return pd.DataFrame(edges)
 
 
def minimum_spanning_tree(returns):
    # finds the load-bearing correlations in the portfolio
    # based on Mantegna (1999) financial network methodology
    corr = returns.corr()
    dist = np.sqrt(2 * (1 - corr))   # distance metric from correlations
    assets = corr.columns.tolist()
    n = len(assets)
 
    # Prim's algorithm for MST
    in_tree = {assets[0]}
    mst_edges = []
 
    while len(in_tree) < n:
        best_edge = None
        best_dist = np.inf
 
        for a in in_tree:
            for b in assets:
                if b in in_tree:
                    continue
                d = dist.loc[a, b]
                if d < best_dist:
                    best_dist = d
                    best_edge = (a, b, round(d, 4),
                                 round(corr.loc[a, b], 4))
 
        if best_edge:
            mst_edges.append(best_edge)
            in_tree.add(best_edge[1])
 
    return pd.DataFrame(mst_edges,
                        columns=['Asset A', 'Asset B',
                                 'Distance', 'Correlation'])
 
 
def stress_transmission(returns, shock_asset, shock_size=-0.05):
    # if IWM drops 5% what does that do to everything else
    # this is what credit risk teams run for concentrated hedge fund exposure
    corr = returns.corr()
    vols = returns.std() * np.sqrt(252)
 
    shock_vol = vols[shock_asset]
    shock_z = shock_size / (shock_vol / np.sqrt(252))   # daily shock in z-score terms
 
    impacts = {}
    for asset in returns.columns:
        if asset == shock_asset:
            impacts[asset] = shock_size
            continue
        beta_to_shock = corr.loc[shock_asset, asset] * (vols[asset] / shock_vol)
        impacts[asset] = round(beta_to_shock * shock_size, 5)
 
    return pd.Series(impacts, name=f'Impact from {shock_asset} {shock_size*100:.1f}%')
 
 
def full_shock_matrix(returns, shock_size=-0.05):
    # run stress_transmission for every asset as the shock source
    results = {}
    for asset in returns.columns:
        results[asset] = stress_transmission(returns, asset, shock_size)
    return pd.DataFrame(results).T