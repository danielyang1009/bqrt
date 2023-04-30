"""
Portfolio Optimization
--------------

Portfolio optimization based on different utility function
"""

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm, trange


def crra_ultility(r, gamma):
    """
    CRRA Utility

    Parameters
    ----------
    r : float
        expected value of return (E[r]), can be predicted value of a factor model
    gamma : float
        risk aversion coefficient

    Returns
    -------
    utility : float
        utility value
    """

    if gamma ==1:
        utility = np.log(r)
    else:
        utility =  (r**(1-gamma))/ (1-gamma)

    return utility


def mean_var_utility(r, sigma, gamma):
    """
    Mean-variance Utility

    Parameters
    ----------
    r : float
        expected value of return (E[r]), can be predicted value of a factor model
    sigma : float
        standard deviation of return
    gamma : float
        risk aversion coefficient

    Returns
    -------
    utility : float
        utility value
    """

    utility = r - 0.5 * gamma * (sigma**2)
    return utility


def two_asset_port_utility(w, ra, rf, sigma=None, gamma=2, utility_func='mean_var'):
    """
    One-period two-asset portfolio optimization, including one risky asset and one risk-free asset. Further step uses `scipy.minimize` to optimized weight based on utility function.

    Parameters
    ----------
    w : float
        optimal weight of risky asset
    ra : float
        risky asset return
    rf : float
        risk-free asset return
    sigma : float, optional
        risky asset standard deviation, by default None
    gamma : int, optional
        risk aversion coefficient, by default 2
    utility_func : string, optional
        choose utility function, by default 'mean_var'

    Returns
    -------
    utility : float
        output negative of utility for minimization (maximize positive utility)
    """

    rp = w * ra + (1 - w) * rf

    # 通过scipy.minimize，寻找w，使得负效用最小（最大化效用）
    if utility_func == 'carra':
        utility = crra_ultility(rp, gamma)

    if utility_func == 'mean_var':
        portfolio_std_dev = w * sigma
        utility = mean_var_utility(rp, portfolio_std_dev, gamma)

    # 最大化效用：最小化负效用
    return -utility


def mean_var_opt_weight(r, sigma, gamma=2, bounds=None):
    """
    Calcuated optimal weight base on mean-variance utility

    Parameters
    ----------
    r : float
        expected value of return
    sigma : float
        risky asset standard deviation, by default None
    gamma : float
        risk aversion coefficient, by defulat 2
    bounds : tuple, optional
        bounds of optimal weight, by default None

    Returns
    -------
    w : float
        optimal weight
    """

    # mean_var的最优权重可以直接通过求导得到，不需要使用最优化
    if bounds:
        assert isinstance(bounds, tuple), 'bounds must be `(lower, upper)` tuple'

    w = r / (gamma * sigma**2)

    if bounds:
        return np.clip([w], bounds[0], bounds[1])[0]
    else:
        return w


def port_optimization(data, yvar_list, scheme, window, *, gamma=3, y_real_suffix=None, y_pred_suffix='_pred', risk_free='rf', bounds=None, freq_adj=1):
    """
    Calcuating optimized two-asset porfolio return, including one risky asset (single risky asset or every risky asset for multiple risky asset case) and one risk-free asset, based on mean-variance utility (`bq.mean_var_opt_weight`)

    Parameters
    ----------
    data : pd.DataFrame
        time-series dataframe contains single or multiple asset real return and predicted return
    yvar_list : str or list
        str or list of asssets(portfolios) name
    scheme : str
        choose between 'rolling' window or 'expanding' window
    window : int
        rolling window size of initial expanding window size
    gamma : float
        risk aversion coefficient
    y_real_suffix : str, optional
        suffix of column name of real asset return, by default None
    y_pred_suffix : str, optional
        suffix of column name of asset return prediction, by default '_pred'
    risk_free : str, optional
        column name of risk-free rate, by default 'rf'
    bounds : tuple, optional
        bounds of optimal weight, by default None
    freq_adj : int, optional
        adjustment of data frequency, daily(252), monthly(12), quarterly(3), yearly(1), by default 1

    Returns
    -------
    pm_tbl : pd.DataFrame
        optimized portfolio return
    """

    assert isinstance(yvar_list, str) or isinstance(yvar_list, list), 'yvar_list需要是str或list'
    if isinstance(yvar_list, str):
        yvar_list = [yvar_list]
    assert risk_free in data.columns, f'{risk_free}不在DataFrame中'

    # 计算投资组合收益需要无风险收益
    pm_tbl = data[[risk_free]].copy()
    data.index.name = 'date'
    data = data.reset_index()

    for yvar in tqdm(yvar_list, desc='y variables loop'):

        # 确保y_real和y_pred在DataFrame中
        y_real = yvar if y_real_suffix == None else yvar + y_real_suffix
        y_pred = yvar + '_pred' if y_pred_suffix == None else yvar + y_pred_suffix
        assert y_real in data.columns, f'{y_real}不在DataFrame中'
        assert y_pred in data.columns, f'{y_pred}不在DataFrame中'

        # 滚动计算最优权重
        s = data.loc[:, ['date', y_real, y_pred]]
        for end in trange(window-1, len(s)-1, leave=False):
            if scheme == 'expanding':
                start = 0
            # index从0开始
            if scheme == 'rolling':
                start = end - (window - 1)

            mu = s.loc[end+1, y_pred]
            # mu_adj = mu * freq_adj # 乘252调整年化
            mu_adj = (1+mu)**freq_adj-1 # 连续复利调整年化
            # s.loc[end+1,'mu'] = mu_adj
            sigma = s.loc[start:end, y_real].std()
            sigma_adj = sigma * np.sqrt(freq_adj)
            # s.loc[end+1,'sigma'] = sigma_adj

            # 注意此时w均为预测值，为至t时刻数据得到
            # 已对齐时间戳为t+1，可以直接与t+1收益相乘
            opt_weight = mean_var_opt_weight(mu_adj, sigma_adj, gamma, bounds)
            s.loc[end+1, yvar+'_w'] = opt_weight

        # 保留yvar_real与yvar_pred列，添加计算后最优化的权重列（yvar_w）
        to_add = s.set_index('date')
        pm_tbl = pd.concat([pm_tbl, to_add], axis='columns')

        # 得到yvar投资组合的最优化收益（yvar_rp）
        # 直接使用赋值报错`DataFrame is highly fragmented`
        port_ret = pm_tbl[yvar+'_w'] * pm_tbl[yvar] + (1-pm_tbl[yvar+'_w']) * pm_tbl[risk_free]
        port_ret.name = yvar+'_rp'
        pm_tbl = pd.concat([pm_tbl, port_ret], axis='columns')

    return pm_tbl


def mean_var_cer(data, yvar_list, *, y_port_suffix='_rp', gamma=3, freq_adj=1):
    """
    Calculate certainty equivalent return (CER) base of mean-variance utility for every asset(portfolio) in `yvar_list`

    Parameters
    ----------
    data : pd.DataFrame
        time-series dataframe of single of multiple portfolio return
    yvar_list : str or list
        str or list of asssets(portfolios) name
    y_port_suffix : str
        suffix of portfolio name
    gamma : float
        risk aversion coefficient
    freq_adj : int, optional
        same as `bq.port_max`

    Returns
    -------
    pd.DataFrame
        result of every portfolios' CER
    """

    if isinstance(yvar_list, str):
        yvar_list = [yvar_list]

    s = pd.DataFrame(index=yvar_list, dtype='float')
    for yvar in yvar_list:

        y_port = yvar + y_port_suffix
        annu_mu = (1+data[y_port].mean())**freq_adj - 1 # 连续复利年化
        # annu_mu = data[yvar+'_rp'].mean()**freq_adj # 乘252年化
        annu_sigma = data[y_port].std() * np.sqrt(freq_adj)

        s.loc[yvar, 'cer'] = mean_var_utility(annu_mu, annu_sigma, gamma)

    return s