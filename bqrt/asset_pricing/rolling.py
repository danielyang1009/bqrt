
"""
Rolling betas
--------------
"""

import pandas as pd
from tqdm.notebook import tqdm


# return index=date, columns=intercept, mkt ...
def sm_rolling_ols(data, yvar, xvar, window=240, interp=True):
    import statsmodels.api as sm
    from statsmodels.regression.rolling import RollingOLS
    
    if interp:
        res = RollingOLS(data[yvar], sm.add_constant(data[xvar]), window=window).fit(params_only=True)
    else:    
        res = RollingOLS(data[yvar], data[xvar], window=window).fit(params_only=True)
        
    return res.params.rename(columns={'const':'intercept'})

# 速度较慢，处理单支股票约50ms
# 处理2800支股票，约4分钟
# return columns=date, symbol, intercept, mkt ...
def rolling_beta(data, time, entity, yvar, xvar, window=240, interp=True, min_time=240):
    entity_list = data[entity].unique()
    
    rolling_betas = pd.DataFrame()
    min_win = window + min_time
    for ent in tqdm(entity_list):
        if data[data[entity] == ent].shape[0] > min_win:
            betas = sm_rolling_ols(data[data[entity] == ent].set_index(time), yvar, xvar, window=window, interp=interp)
            betas[entity] = ent # mark
            rolling_betas = pd.concat([rolling_betas, betas], axis=0)

    rolling_betas = rolling_betas.dropna().reset_index() # get date column back
    rolling_betas = rolling_betas.sort_values(['symbol','date']).reset_index(drop=True)
    
    if interp:
        return rolling_betas[[time]+[entity]+['intercept']+xvar]
    else:
        return rolling_betas[[time]+[entity]+xvar]