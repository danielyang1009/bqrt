"""
Common
------

Common utility tools
"""

import pandas as pd
import numpy as np


def describe(df:pd.DataFrame, *, params_digi:int=4, tstat_digi:int=2):
    """
    Create pandas describe like dataframe, contains skewness and kurtosis, t-statistics and p-value. Only select columns with float dtype.

    Only describe columns with float dtype. If a column contains all NaNs, then result is NaNs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame object to be described, exclude columns not to be described
    digits : int
        number of digits to show

    Returns
    -------
    result : pd.DataFrame
        DataFrame stats
    """
    import scipy.stats as stats

    result_list = []
    params_format = '{{:.{}f}}'.format(params_digi)
    tstat_format = '{{:.{}f}}'.format(tstat_digi)

    # deal with pd.Series
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # only include `float` dtype
    df = df.select_dtypes(include='float')

    # columns contain all NaNs value
    not_all_nan_cols = df.columns[~df.isnull().all()].to_list()

    for col in df.columns.to_list():
        des = stats.describe(df[col], nan_policy='omit')

        if col in not_all_nan_cols:
            ttest = stats.ttest_1samp(df[col],0,nan_policy='omit')
            parts = {
                'Count': '{}'.format(int(des.nobs)),
                'Mean': params_format.format(des.mean),
                'SD': params_format.format(np.sqrt(des.variance)),
                # 'var': params_format.format(des.variance),
                'Min': params_format.format(des.minmax[0]),
                '1%': params_format.format(df[col].quantile(0.01)),
                '5%': params_format.format(df[col].quantile(0.05)),
                '25%': params_format.format(df[col].quantile(0.25)),
                '50%': params_format.format(df[col].quantile(0.50)),
                '75%': params_format.format(df[col].quantile(0.75)),
                '95%': params_format.format(df[col].quantile(0.95)),
                '99%': params_format.format(df[col].quantile(0.99)),
                'Max': params_format.format(des.minmax[1]),
                'Skew': params_format.format(des.skewness),
                'Kurt': params_format.format(des.kurtosis),
                'T-stat': tstat_format.format(ttest[0]),
                'P-val': tstat_format.format(ttest[1])
            }
            result_list.append(parts)
        else:
            parts = {
                'Count': '{}'.format(int(des.nobs)),
                'Mean': np.NaN, 'SD': np.NaN, 'Min': np.NaN, '5%': np.NaN, '25%': np.NaN, '50%': np.NaN, '75%': np.NaN,
                '95%': np.NaN, 'Max': np.NaN, 'Skew': np.NaN, 'Kurt': np.NaN, 'T-stat': np.NaN, 'P-val':np.NaN,
            }
            result_list.append(parts)

    result = pd.DataFrame(result_list).T
    result.columns = df.columns
    return result.T.fillna('')


def check_ptable(data, *, threshold=None, star=False, columns=None, count=False, digi=None):
    """
    Check p-value table

    Parameters
    ----------
    data : pd.DataFrame
        contains only p-values
    threshold : float, optional
        show only p-values below threshold, by default None
    star : bool, optional
        add stars to final result, by default False
    columns : list, optional
        choose columns with p-value, by default None
    count : float, optional
        count total significant level, by default False
    digi : int, optional
        round to decimal
    Returns
    -------
    result : pd.DataFrame
        return result of original table format, with `columns` modified, and output format is string
    """

    assert isinstance(data, pd.DataFrame), '应为DataFrame'
    if columns != None:
        assert isinstance(columns,str) or isinstance(columns,list), 'columns 必须为str或list'
    if isinstance(columns, str):
        columns = [columns]
    # 如果未指定columns检查所有columns为float，指定则只检查指定列
    if columns == None:
        columns = data.columns
    assert all(data[columns].dtypes == float), '所有的列应为float'

    # float p值部分
    ptable = data[columns]
    # 是否统计显著数目
    if count:
        ttl_count = ptable.count().sum()
        for lower, upper, stars in zip([0,0.01,0.05],[0.01,0.05,0.1],['***','**','*']):
            # print(lower,upper,stars)
            sig_count = ptable[(ptable>lower) & (ptable<=upper)].count().sum()
            print('{:>3}: {}/{}'.format(stars, sig_count, ttl_count))

    # 是否仅显示小于threshold
    if threshold != None:
        ptable = ptable[ptable <= threshold]

    # 是否打星，将结果转换为str
    # 注意问题：如0.103在数字只有**，但一旦round(2)之后变为0.1那么就可能是***
    # 不能先round 再打星
    if star == True and digi == None:
        ptable = ptable.applymap(lambda x: str(x)+''.join(['*' for t in [.01, .05, .1] if x<=t]))

    if star == True and digi != None:
        ptable = ptable.applymap(lambda x: '{:.{}f}'.format(x,digi) + ''.join(['*' for t in [.01, .05, .1] if x<=t]))

    if digi != None:
        ptable = ptable.round(digi)

    # 剔除nan，保持原格式输出
    ptable = ptable.astype(str).replace({'nan':''})
    result = data.copy()
    for col in columns:
        result[col] = ptable[col]
    return result


# TODO
def cumulative_ret(df:pd.DataFrame, step_forward, shift_back=False):
    # 计算累计收益
    # 宽表，列为资产日月收益率，计算多期的累计收益

    # 问题在于是否考虑当天作为累计收益的一部分（累计收益不考虑当天，与预测回归逻辑一致）
    # 在预测回归中X为当天，Y为明天
    # 那么对于累计收益，应该从明天开始计算，若3天累计收益
    # 今天的X，对应明天后天大后天的Y，此时X对应shift(2)

    # 计算多少天的累计收益，3天，则rolling(3)
    # 若step_forward =1??? 就是预测回归？？？，原表不变
    cum_ret = (df+1).rolling(step_forward).apply(np.prod)-1

    if shift_back == True:
        # shift_back之后，默认对其X日期（预测回归），X无需再进行调整，可直接回归
        # 即若3天累计收益，Y计算t,t+1,t+2三天，数据最终位置为t+2
        # shift_back 3天后，最终位置为t-1，直接与X进行回归（预测回归）
        cum_ret = cum_ret.shift(-step_forward)
    else:
        # 不进行调整
        # 若在预测模型中X已shift(1)
        # 则Y.shift(-step_forward+1)进行预测回归
        return cum_ret


def show_all(df:pd.DataFrame):
    from IPython.display import display
    with pd.option_context('display.max_rows', 1000, 'display.max_columns', 100):
        display(df)


def cp_plot(df:pd.DataFrame, figsize=(8,4)):
    """
    Plot cumprod

    Parameters
    ----------
    df : pd.DataFrame
        _description_
    figsize : tuple, optional
        _description_, by default (8,4)
    """

    (df+1).cumprod().plot(figsize=figsize)


def first_notna_index(df:pd.DataFrame):
    """print first row index that is notna

    Parameters
    ----------
    df : pd.DataFrame
        input DataFrame

    Returns
    -------
    pd.Series
        return index of first notna row
    """
    return df.apply(lambda col: col.first_valid_index())