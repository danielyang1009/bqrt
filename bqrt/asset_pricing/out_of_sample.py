"""
Out-of-sample Predictability
----------------------------
"""

import numpy as np
import pandas as pd

def pred_reg(data, yvar_list, xvar_list, scheme, window, *, benchmark=None, intercept=True, min_data_count=None):
    """
    Prdictive regression using rolling/expanding window. Different schemes include 'mean' (historical mean with same window), 'zero', and custom input X predictors list (must be).

    Important
    ---------
    `data` DataFrame must be organized with lead-lag format beforehand, e.g. with some date t+1, input data y(t+1) and x(t) in same row. Prediction result of t+1 based on initial data from 0 to t (rolling or expanding).

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
        choose between 'mean', or 'zero', or list of X variables for creating benchmark predicting model
    intercept : bool, optional
        whether including intercept in predictive regression, by default 'True'
    min_data_count : int, optional
        minimum number of data points that both y and X contains for regression, otherwise, will trigger 'zero-size array to reduction operation maximum which has no identity', e.g. y contains all NaNs at the beginning of data set (first rolling window), however, X contins all data points

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
    assert benchmark in [None,'zero','mean'] or isinstance(benchmark, list), "benchmark只能选择'mean'，'zero'，或输入x变量列表"
    # assert len(benchmark) > 0, "benchmark，X列表为空"

    from statsmodels.api import OLS, add_constant
    from tqdm.notebook import tqdm, trange

    if isinstance(yvar_list, str):
        yvar_list = [yvar_list]

    # 方便计算，使用`s.loc[start,end]`数值进行窗口变动调整
    data.index.name = 'date'

    predition = pd.DataFrame(dtype='float')
    for yvar in tqdm(yvar_list, desc='y variables loop'):

        # 使用数字而非日期对df进行切片
        s = data.loc[:,[yvar]+xvar_list].copy().reset_index()
        if intercept:
            s = add_constant(s)

        # 需要注意s[start:end]不包含end行，s.loc[start:end]根据index，包含end行
        # 需注意，虽然整体样本可能可以进行，但滚动起来时，还需要判断滚动窗口内的样本量
        for end in trange(window-1, len(s)-1, leave=False):
            if scheme == 'expanding':
                start = 0
            # index从0开始
            if scheme == 'rolling':
                start = end - (window - 1)

            y_train = s.loc[start:end, yvar]
            if intercept:
                X_train = s.loc[start:end, ['const']+xvar_list]
                X_test = s.loc[end+1, ['const']+xvar_list]
            else:
                X_train = s.loc[start:end, xvar_list]
                X_test = s.loc[end+1, xvar_list]

            # y_train为单变量
            y_train_quilified = y_train.notna().sum() > min_data_count
            # X_train中为多变量
            X_train_quilified = all(X_train.notna().sum() > min_data_count)

            # 判断数据是否男足min_data_count所要求的数据数量，以进行回归
            if y_train_quilified and X_train_quilified:

                # 预测回归，领先滞后回归
                reg_pred = OLS(y_train, X_train, missing='drop').fit()
                # 当y输入为超额收益时，预测结果也为超额收益
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
                        if intercept:
                            X_train_bm = s.loc[start:end, ['const']+benchmark]
                            X_test_bm = s.loc[end+1, ['const']+benchmark]
                        else:
                            X_train_bm = s.loc[start:end, benchmark]
                            X_test_bm = s.loc[end+1, benchmark]

                        reg_bench = OLS(y_train, X_train_bm, missing='drop').fit()
                        y_pred_bm = reg_bench.predict(X_test_bm)
                        s.loc[end+1, yvar+'_bench'] = y_pred_bm[0]

        s = s.set_index('date')
        to_add = s.loc[:,s.columns.str.contains(yvar)]
        predition = pd.concat([predition,to_add], axis='columns')

    return predition


def oos_r2(data, yvar_list, y_real_suffix=None, y_pred_suffix='_pred', y_bench_suffix='_bench'):

    assert isinstance(yvar_list, str) or isinstance(yvar_list, list), 'yvar_list需要是str或list'
    if isinstance(yvar_list, str):
        yvar_list = [yvar_list]

    r2_table = pd.DataFrame(index=yvar_list, columns=['oos_r2'], dtype='float')
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
    # 直接dropna，由于有的没有数据，因此会将大量数据删除，需要单独计算！！！
    # s = s.dropna()
    # 每一列为yvar的计算后统计值值的时间序列
    cw_tbl = pd.DataFrame(dtype='float')
    cw_tbl.index.name = 'port'
    for yvar in yvar_list:

        y_real = yvar if y_real_suffix == None else yvar + y_real_suffix
        y_pred = yvar + y_pred_suffix
        y_bench = yvar + y_bench_suffix

        # calculating MSPE statistic
        to_add = (s[y_real]- s[y_bench])**2 - (s[y_real] - s[y_pred])**2 + (s[y_bench] - s[y_pred])**2
        to_add.name = yvar
        cw_tbl = pd.concat([cw_tbl, to_add], axis='columns')

    return cw_tbl


def cw_table(cw_tbl, one_side=True):
    """
    Calcuate Clark & West (2007) t statistic, based on `cw_stat` result. Based on Bakshi.etal (2010), Rapach.etal (2016) etc, performs one-side t test by default.

    Parameters
    ----------
    cw_tbl : pd.DataFrame
        DataFrame with DatetimeIndex (time-series) and assets' CW statistic in columns, result from `bq.cw_stat`

    Returns
    -------
    cw_sig : pd.DataFrame
        DataFrame of t-value and p-value of CW statistic
    """
    import statsmodels.formula.api as smf

    # 根据bakshi，应该考虑单边p值？即将双边的P值除2即可？
    cw_sig = pd.DataFrame(index=cw_tbl.columns, columns=['tval','pval'], dtype='float')
    for yvar in cw_tbl.columns:
        reg = smf.ols('{} ~ 1'.format(yvar), data=cw_tbl).fit()
        # print(reg.tvalues[0])
        # print(type(reg.tvalues[0]))
        cw_sig.loc[yvar,'tval'] = reg.tvalues[0]
        if one_side == True:
            cw_sig.loc[yvar,'pval'] = reg.pvalues[0] / 2
        else:
            cw_sig.loc[yvar,'pval'] = reg.pvalues[0]

    return cw_sig




