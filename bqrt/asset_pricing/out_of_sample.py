"""
Out-of-sample Predictability
----------------------------
"""

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm, trange


def singl_pred(data, yvar, xvar_list, scheme, window, benchmark=None):

    import statsmodels.api as sm
    # window 为训练样本大小（rolling window），或expanding window起始大小
    assert isinstance(data.index, pd.DatetimeIndex), 'index 为DatetimeIndex'
    assert isinstance(window, int), '窗口大小应为int'
    assert window < len(data), '窗口大于数据样本量'
    assert scheme in ['expanding','rolling'], 'scheme只能选择expanding和rolling两种方式'
    assert benchmark in ['zero','mean'] or isinstance(benchmark, list), "benchmark只能选择'mean'，'zero'，或输入x变量列表"
    # window 为 expanding起始大小，或rolling滚动窗口大小

    # 方便计算，使用`s.loc[start,end]`数值进行窗口变动调整
    data.index.name = 'date'
    s = data.copy().reset_index()

    for end in trange(window-1, len(s)-1, leave=False):
        if scheme == 'expanding':
            start = 0
        # index从0开始
        if scheme == 'rolling':
            start = end - (window - 1)

        # 需要注意s[start:end]不包含end行，s.loc[start:end]根据index，包含end行
        # 剔除window中的最后一条作为prediction
        # print(start, end)
        X_train = s.loc[start:end, xvar_list]
        y_train = s.loc[start:end, yvar]
        reg_pred = sm.OLS(y_train, X_train, missing='drop').fit()

        # 预测回归
        X_test = s.loc[end+1, xvar_list]
        # 预测为超额收益
        y_pred = reg_pred.predict(X_test)
        s.loc[end+1, yvar+'_pred'] = y_pred[0]

        # 可以选择不计算benchmark
        if benchmark:
            # 不同的benchmark
            if benchmark == 'mean':
                # 为历史收益（非超额）均值
                s.loc[end+1, yvar+'_bench'] = s.loc[start:end, yvar].mean()

            if benchmark == 'zero':
                s.loc[end+1, yvar+'_bench'] = 0

            if isinstance(benchmark, list):
                X_train_bm = s.loc[start:end, benchmark]
                reg_bench = sm.OLS(y_train, X_train_bm, missing='drop').fit()
                y_pred_bm = reg_bench.predict(X_test)
                s.loc[end+1, yvar+'_bench'] = y_pred_bm[0]

    s = s.set_index('date')
    return s.loc[:,s.columns.str.contains(yvar)]


def multi_yvar_pred(data, yvar_list, xvar_list, scheme, window, benchmark=None):

    assert isinstance(yvar_list, str) or isinstance(yvar_list, list), 'yvar_list需要是str或list'
    if isinstance(yvar_list, str):
        yvar_list = [yvar_list]

    s = pd.DataFrame()
    for yvar in tqdm(yvar_list, desc='y variables loop'):
        to_add = singl_pred(data, yvar, xvar_list, scheme, window, benchmark)
        s = pd.concat([s,to_add], axis='columns')

    return s


def prediction(data, yvar_list, xvar_list, scheme, window, benchmark=None):
    """
    Prdiction using rolling/expanding window. Different schemes include 'mean' (historical mean with same window), 'zero', and custom input X predictors list (must be).

    `data` DataFrame must be organized with lead-lag format beforehand, e.g. with some date t, y(t) and x(t-1) in same row.

    Parameters
    ----------
    data : pd.DataFrame
        DatetimeIndex as index, columns including leading Y variables, and lagging X variables
    yvar_list : str or list
        str (single) / list(multiple) of Y variable(s)
    xvar_list : list
        list of X variables
    scheme :
        rolling window or expanding window
    window : int
        size sample data used (training sample size) of rolling window or initial expanding window
    benchmark : str, optional
        default is 'mean', or 'zero', or list of X variables for creating benchmark predicting model

    Returns
    -------
    prediction : pd.DataFrame
        prediction result
    """

    assert isinstance(yvar_list, str) or isinstance(yvar_list, list), 'yvar_list需要是str或list'
    # window 为训练样本大小（rolling window），或expanding window起始大小
    assert isinstance(data.index, pd.DatetimeIndex), 'index 为DatetimeIndex'
    assert isinstance(window, int), '窗口大小应为int'
    assert window < len(data), '窗口大于数据样本量'
    assert scheme in ['expanding','rolling'], 'scheme只能选择expanding和rolling两种方式'
    # assert benchmark in ['zero','mean'] or isinstance(benchmark, list), "benchmark只能选择'mean'，'zero'，或输入x变量列表"
    # window 为 expanding起始大小，或rolling滚动窗口大小

    import statsmodels.api as sm

    if isinstance(yvar_list, str):
        yvar_list = [yvar_list]

    # 方便计算，使用`s.loc[start,end]`数值进行窗口变动调整
    data.index.name = 'date'

    predition = pd.DataFrame()
    for yvar in tqdm(yvar_list, desc='y variables loop'):

        s = data.loc[:,[yvar]+xvar_list].copy().reset_index()
        for end in trange(window-1, len(s)-1, leave=False):
            if scheme == 'expanding':
                start = 0
            # index从0开始
            if scheme == 'rolling':
                start = end - (window - 1)

            # 需要注意s[start:end]不包含end行，s.loc[start:end]根据index，包含end行
            # 剔除window中的最后一条作为prediction
            # print(start, end)
            X_train = s.loc[start:end, xvar_list]
            y_train = s.loc[start:end, yvar]
            reg_pred = sm.OLS(y_train, X_train, missing='drop').fit()

            # 预测回归
            X_test = s.loc[end+1, xvar_list]
            # 预测为超额收益
            y_pred = reg_pred.predict(X_test)
            s.loc[end+1, yvar+'_pred'] = y_pred[0]

            # 可以选择不计算benchmark
            if benchmark is not None:
                # 不同的benchmark
                if benchmark == 'mean':
                    # 为历史收益（非超额）均值
                    s.loc[end+1, yvar+'_bench'] = s.loc[start:end, yvar].mean()

                if benchmark == 'zero':
                    s.loc[end+1, yvar+'_bench'] = 0

                if isinstance(benchmark, list):
                    X_train_bm = s.loc[start:end, benchmark]
                    reg_bench = sm.OLS(y_train, X_train_bm, missing='drop').fit()
                    y_pred_bm = reg_bench.predict(X_test)
                    s.loc[end+1, yvar+'_bench'] = y_pred_bm[0]

        s = s.set_index('date')
        to_add = s.loc[:,s.columns.str.contains(yvar)]
        predition = pd.concat([predition,to_add], axis='columns')

    return predition


def oos_r2(data, yvar_list, y_real_suffix=None, y_pred_suffix='_pred', y_bench_suffix='_bench'):

    assert isinstance(yvar_list, str) or isinstance(yvar_list, list), 'yvar_list需要是str或list'
    if isinstance(yvar_list, str):
        yvar_list = [yvar_list]

    r2_table = pd.DataFrame(index=yvar_list, columns=['oos_r2'])
    r2_table.index.name = 'port'
    for yvar in yvar_list:

        y_real = yvar if y_real_suffix == None else yvar + y_real_suffix
        y_pred = yvar + y_pred_suffix
        y_bench = yvar + y_bench_suffix

        assert y_real in data.columns, f'{y_real}不在DataFrame中'
        assert y_pred in data.columns, f'{y_pred}不在DataFrame中'
        assert y_bench in data.columns, f'{y_bench}不在DataFrame中'

        s = data[[y_real, y_pred, y_bench]].dropna().copy()
        ss_pred = np.sum((s[y_real] - s[y_pred])**2)
        ss_bench = np.sum((s[y_real] - s[y_bench])**2)

        r2_table.loc[yvar,'oos_r2'] = 1 - ss_pred / ss_bench

    return r2_table


# 根据CW2007以及Bakshi.et.at2010，MSPE-adjusted statistic
def cw_stat(s, yvar_list, *, y_real_suffix=None, y_pred_suffix='_pred', y_bench_suffix='_bench'):

    assert isinstance(yvar_list, str) or isinstance(yvar_list, list), 'yvar_list需要是str或list'
    if isinstance(yvar_list, str):
        yvar_list = [yvar_list]

    # 由sing_pred或mult_pred结果进行计算
    s = s.dropna()
    # 每一列为yvar的计算后统计值值的时间序列
    cw_tbl = pd.DataFrame()
    cw_tbl.index.name = 'port'
    for yvar in yvar_list:

        y_real = yvar if y_real_suffix == None else yvar + y_real_suffix
        y_pred = yvar + y_pred_suffix
        y_bench = yvar + y_bench_suffix

        # calculating MSPE statistic
        to_add = (s[y_real]- s[y_bench])**2 - (s[y_real]- s[y_pred])**2 + (s[y_bench] - s[y_pred])**2
        to_add.name = yvar
        cw_tbl = pd.concat([cw_tbl, to_add], axis='columns')
    return cw_tbl


def cw_t(f_tbl):
    """
    Calcuate Clark & West (2007) t statistic, based on `cw_stat` result

    Parameters
    ----------
    f_tbl : pd.DataFrame
        DataFrame with DatetimeIndex (time-series) and assets' CW statistic in columns
    """
    import statsmodels.formula.api as smf

    # 根据cw_stat的时间序列计算对应的t值
    # 根据bakshi，应该考虑单边p值？即将双边的P值除2即可？
    for yvar in f_tbl.columns:
        reg = smf.ols('{} ~ 1'.format(yvar), data=f_tbl).fit()
        f_tbl.loc[yvar,'tval'] = reg.tvalues[0]
        f_tbl.loc[yvar,'pval'] = reg.pvalues[0]

