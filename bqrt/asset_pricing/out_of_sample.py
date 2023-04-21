"""
Out-of-sample Predictability
----------------------------
"""

import numpy as np
import pandas as pd
from tqdm.notebook import trange, tqdm


def singl_pred(data, yvar, xvar_list, scheme, window, benchmark='mean'):
    """
    Prdiction using rolling/expanding window. Different schemes include 'mean' (historical mean with same window), 'zero', and custom input X predictors list (must be).

    `data` DataFrame must be organized with lead-lag format beforehand, e.g. with some date t, y(t) and x(t-1) in same row.

    Parameters
    ----------
    data : pd.DataFrame
        DatetimeIndex as index, columns including leading Y variables, and lagging X variables
    yvar : str
        single string of Y variable
    xvar_list : list
        list of X variables
    scheme :
        rolling window or expanding window
    window : int
        size sample data used (training sample size) of rolling window or initial expanding window
    benchmark : str, optional
        default is 'mean', or 'zero', or list of X variables for creating benchmark
    """
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

#         需要注意s[start:end]不包含end行，s.loc[start:end]根据index，包含end行
#         剔除window中的最后一条作为prediction
#         print(start, end)
        X_train = s.loc[start:end, xvar_list]
        y_train = s.loc[start:end, yvar]
        reg_pred = sm.OLS(y_train, X_train, missing='drop').fit()

#         预测回归
        X_test = s.loc[end+1, xvar_list]
        y_pred = reg_pred.predict(X_test)
        s.loc[end+1, yvar+'_pred'] = y_pred[0]

#         不同的benchmark
        if benchmark == 'mean':
            s.loc[end+1, yvar+'_bench'] = s.loc[start:end, yvar].mean()

        if benchmark == 'zero':
            s.loc[end+1, yvar+'_bench'] = 0

        if isinstance(benchmark,list):
            X_train_bm = s.loc[start:end, benchmark]
            reg_bench = sm.OLS(y_train, X_train_bm, missing='drop').fit()
            y_pred_bm = reg_bench.predict(X_test)
            s.loc[end+1, yvar+'_bench'] = y_pred_bm[0]

    s = s.set_index('date')
    return s.loc[:,s.columns.str.contains(yvar)]


def multi_yvar_pred(data, yvar_list, xvar_list, scheme, window, benchmark='mean'):

    assert isinstance(yvar_list, list), 'yvar_list需要是list'


    if len(yvar_list) == 1:
        s = singl_pred(data, yvar_list[0], xvar_list, scheme, window, benchmark)

    if len(yvar_list) > 1:
        s = pd.DataFrame()
        for yvar in tqdm(yvar_list, desc='y variables loop'):
            to_add = singl_pred(data, yvar, xvar_list, scheme, window, benchmark)
            s = pd.concat([s,to_add], axis='columns')

    return s


def oos_r2(s, y, y_pred, y_bench):

    # na不会影响计算
    s = s[[y, y_pred, y_bench]].dropna()

    ss_pred = np.sum((s[y] - s[y_pred])**2)
    ss_bench = np.sum((s[y] - s[y_bench])**2)
    # print(ss_pred, ss_bench)

    oos_r2 = 1 - ss_pred / ss_bench
    # critifal value
    # oos_f = len(df) * ss_bench / ss_pred * oos_r2

    return oos_r2


def multi_oos_r2(s, yvar_list):

    assert isinstance(yvar_list, list), 'yvar_list需要是list'

    r2_table = pd.DataFrame(index=yvar_list, columns=['r2'])
    r2_table.index.name = 'port'
    for yvar in yvar_list:
        r2_table.loc[yvar,'r2'] =oos_r2(s, yvar, yvar+'_pred', yvar+'_bench')

    return r2_table


# 根据CW2007以及Bakshi.et.at2010
def mspef(s, yvar_list):

    # 由sing_pred或mult_pred结果进行计算
    s = s.dropna()
    # 每一列为yvar的计算后f值的时间序列
    f_tbl = pd.DataFrame()
    f_tbl.index.name = 'port'
    for yvar in yvar_list:
        # 为列
        f_to_add = (s[yvar]- s[yvar+'_bench'])**2 - (s[yvar]- s[yvar+'_pred'])**2 + (s[yvar+'_bench'] - s[yvar+'_pred'])**2
        f_to_add.name = yvar
        f_tbl = pd.concat([f_tbl, f_to_add], axis='columns')
    return f_tbl


def mspe_t(f_tbl):
    import statsmodels.formula.api as smf

    # 根据f的时间序列计算对应的t值
    # 根据bakshi，应该考虑单边p值？即将双边的P值除2即可？
    for yvar in f_tbl.columns:
        reg = smf.ols('{} ~ 1'.format(yvar), data=f_tbl).fit()
        reg = smf.ols('{} ~ 1'.format(yvar), data=f_tbl).fit()
        f_tbl.loc[yvar,'tval'] = reg.tvalues[0]
        f_tbl.loc[yvar,'pval'] = reg.pvalues[0]

