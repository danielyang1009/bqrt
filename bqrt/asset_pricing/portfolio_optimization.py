"""
Portfolio Optimization
--------------
"""

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm, trange


def crra_ultility(r, gamma):

    if gamma ==1:
        utility = np.log(r)
    else:
        utility =  (r**(1-gamma))/ (1-gamma)

    return utility


def mean_var_utility(r, sigma, gamma):

    utility = r - 0.5 * gamma * (sigma**2)
    return utility


def two_asset_port_utility(w, ra, rf, sigma=None, gamma=2, utility_func='mean_var'):
    """
    供`scipy.minimize`计算最优权重

    Parameters
    ----------
    w : float
        optimal weight of risky asset
    ra : float
        risky asset return
    rf : float
        risk-free asset return
    sigma : risky asset standard deviation, optional
        _description_, by default None
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


def mean_var_opt_weight(r, sigma, gamma, bounds=None):

    # mean_var的最优权重可以直接通过求导得到，不需要使用最优化
    if bounds:
        assert isinstance(bounds, tuple), 'bounds must be `(lower, upper)` tuple'

    w = r / (gamma * sigma**2)

    if bounds:
        return np.clip([w], bounds[0], bounds[1])[0]
    else:
        return w


def port_max(data, yvar, scheme, window, gamma, *, y_real=None, y_pred=None, bounds=None, freq_adj=None):

    # y_pred为预测收益 E(r_excess + r_f)
    data.index.name = 'date'
    s = data.copy().reset_index()

    for end in trange(window-1, len(s)-1, leave=False):
        if scheme == 'expanding':
            start = 0
        # index从0开始
        if scheme == 'rolling':
            start = end - (window - 1)

        # 未指定y_real则为yvar自身
        if y_real == None:
            y_real = yvar
        # 未指定，则默认为yvar+'_pred‘
        if y_pred == None:
            y_pred = yvar + '_pred'

        mu = s.loc[end+1, y_pred]
        # mu_adj = mu * freq_adj # 乘252调整年化
        mu_adj = (1+mu)**freq_adj-1 # 连续复利调整年化
        # s.loc[end+1,'mu'] = mu_adj
        sigma = s.loc[start:end, y_real].std()
        sigma_adj = sigma * np.sqrt(freq_adj)
        # s.loc[end+1,'sigma'] = sigma_adj

        # 注意此时w均为预测值，为至t时刻数据得到
        # 已对其时间戳，可以直接与t+1收益相乘
        opt_weight = mean_var_opt_weight(mu_adj, sigma_adj, gamma, bounds)
        s.loc[end+1, yvar+'_w'] = opt_weight

    s = s.set_index('date')
    return s.loc[:,[y_real, y_pred, yvar+'_w']]


# 根据yvar_list中的各个yvar的未来估计值yvar_pred计算最优化权重，而后通过实际收益yvar_real(yar)计算实际两资产投资组合收益(yvar_rp)
def multi_port_max(data, yvar_list, scheme, window, gamma, *, y_real=None, y_pred=None, risk_free=None, bounds=None, freq_adj=None):


    assert isinstance(yvar_list, str) or isinstance(yvar_list, list), 'yvar_list需要是str或list'

    if isinstance(yvar_list, str):
        yvar_list = [yvar_list]

    if not risk_free:
        risk_free = 'rf'

    s = data[[risk_free]].copy()
    for yvar in tqdm(yvar_list, desc='y variables loop'):

        to_add = port_max(data, yvar, scheme, window, gamma, y_real=y_real, y_pred=y_pred, bounds=bounds, freq_adj=freq_adj)
        s = pd.concat([s,to_add], axis='columns')

        # 计算投资组合收益
        s[yvar+'_rp'] = s[yvar+'_w'] * s[yvar] + (1-s[yvar+'_w']) * s[risk_free]

    return s


def mean_var_cer(data, yvar_list, gamma, freq_adj=None):

    if isinstance(yvar_list, str):
        yvar_list = [yvar_list]

    s = pd.DataFrame(index=yvar_list)
    for yvar in yvar_list:
        annu_mu = (1+data[yvar+'_rp'].mean())**freq_adj - 1 # 连续复利年化
        # annu_mu = data[yvar+'_rp'].mean()**freq_adj # 乘252年化
        annu_sigma = data[yvar+'_rp'].std() * np.sqrt(freq_adj)

        s.loc[yvar, 'cer'] = mean_var_utility(annu_mu, annu_sigma, gamma)

    return s