"""
Out-of-sample Predictability
----------------------------
"""

import numpy as np
import pandas as pd
from tqdm.notebook import trange, tqdm


def singl_pred(data, yvar, xvar_list, scheme, window, benchmark='mean'):

    import statsmodels.api as sm

    # window 为训练样本大小（rolling window），或expanding window起始大小
    assert isinstance(data.index, pd.DatetimeIndex), 'index 为DatetimeIndex'
    assert window < len(data), '窗口大于数据样本量'
    assert scheme in ['expanding','rolling'], 'scheme只能选择expanding和rolling两种方式'
    assert benchmark in ['zero','mean'] or isinstance(benchmark, list), "benchmark只能选择'mean'，'zero'，或输入x变量列表"
    # window 为 expanding起始大小，或rolling滚动窗口大小

    # 方便计算
    data.index.name = 'date'
    df = data.copy().reset_index()

    for end in trange(window-1, len(df)-1, leave=False):
        if scheme == 'expanding':
            start = 0
        # index从0开始
        if scheme == 'rolling':
            start = end - (window - 1)

#         需要注意df[start:end]不包含end行，df.loc[start:end]根据index，包含end行
#         剔除window中的最后一条作为prediction
#         print(start, end)
        X_train = df.loc[start:end, xvar_list]
        y_train = df.loc[start:end, yvar]
        reg_pred = sm.OLS(y_train, X_train, missing='drop').fit()

#         预测回归
        X_test = df.loc[end+1, xvar_list]
        y_pred = reg_pred.predict(X_test)
        df.loc[end+1, yvar+'_pred'] = y_pred[0]

#         不同的benchmark
        if benchmark == 'mean':
            df.loc[end+1, yvar+'_bench'] = df.loc[start:end, yvar].mean()

        if benchmark == 'zero':
            df.loc[end+1, yvar+'_bench'] = 0

        if isinstance(benchmark,list):
            X_train_bm = df.loc[start:end, benchmark]
            reg_bench = sm.OLS(y_train, X_train_bm, missing='drop').fit()
            y_pred_bm = reg_bench.predict(X_test)
            df.loc[end+1, yvar+'_bench'] = y_pred_bm[0]

    df = df.set_index('date')
    return df.loc[:,df.columns.str.contains(yvar)]


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

    for yvar in yvar_list:
        print(yvar,oos_r2(s, yvar, yvar+'_pred', yvar+'_bench'))