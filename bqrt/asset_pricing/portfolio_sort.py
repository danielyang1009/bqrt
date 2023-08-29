"""
Portfolio Sort
--------------

目前对于无法qcut的处理方法是直接略过该组，全部标记为nan，并打印该组
"""

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm


def port_vw_return(mv_raw, ret_raw):
    """
    Calculate value-weighted portfolio return. All inputs are in format of wide-table(date as index, individual asset as columns).

    Parameters
    ----------
    mv_raw : pd.DataFrame
        market value dataframe, used to calcuate weight of each individual stock
    ret_raw : pd.DataFrame
        return dataframe

    Returns
    -------
    pd.DataFrame
        time-series of value-weighted portfolio return
    """
    # assertion mv_df, ret_df
    mv_df, ret_df = mv_raw.copy(), ret_raw.copy()
    assert isinstance(mv_df, pd.DataFrame), 'mv_df应为DataFrame'
    assert isinstance(mv_df, pd.DataFrame), 'ret_df应为DataFrame'
    # 确保有相同的行与列
    assert all(mv_df.columns == ret_df.columns), '列不一致'
    assert all(mv_df.index == ret_df.index), '行不一致'

    # 计算每天总市值
    ttl_mv_by_row = mv_df.sum(axis='columns')
    # 计算权重 = 市值/总市值
    mv_df = mv_df.div(ttl_mv_by_row, axis='rows')

    result = mv_df.mul(ret_df)
    # 若该行全部为NaN，那么求和将得到0.00，不能直接使用`df.sum(axis='columns')`
    # 当这行有部分na，则照常累加；若全部都是na，代表数据缺失，则应返回na
    return result.apply(lambda x: np.nan if x.isna().all() else x.sum(), axis='columns')


def qcut_wrapper(data, bins_list, lables):
    """
    Wrapper of `pd.qcut`, in case of ValueError, print out that row of data, drop

    Parameters
    ----------
    data_list : pd.Series or list
        list for qcut
    b_list : list
        list of to create unequal sample size bins,in format of `[30, 40, 30]`, by default None
    lables : list
        how bins are labeled

    Returns
    -------
    result : pd.Series
        labeled result of quct
    """

    assert isinstance(data, pd.Series) | isinstance(data, list), 'x must be pd.Series or list'

    try:
        result = pd.qcut(data, bins_list, lables)
    except ValueError:
        print(data)
        result = [np.nan]*len(data)

    return result


def qcut_debug(data, bins_list):
    """
    Try if `pd.qcut` can be performed on time-series of data (DatetimeIndex)

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with DatetimeIndex
    b_list : list
        list of to create unequal sample size bins,in format of `[30, 40, 30]`, by default None
    """
    for date in data.index:
        try:
            pd.qcut(data.loc[date], bins_list)
        except ValueError:
            print(date)


def portfolio_sort(sig_raw, ret_raw, scheme, b_num=None, b_list=None):
    """
    Calculating 1 step forward portfolios return based on signal dataframe, by different scheme methods. All inputs are in format of wide-table (date as index, individual asset as columns).

    注意：
    - 【手动】计算前需要确保每行有足够的数据进行分组

    计算过程，两表日期表相同的前提下（index为时间，columns为品种）：
    1. 根据收益率表，通过`shift(1)`指标表，对指标表建立一个mask（后续转化为权重表）
        - 确保：指标表（权重表）的后一步，一定存在收益数据
        - 当t指标表有值，t+1收益表无值：根据收益表mask，去除指标表原值，不参与计算
        - 当t指标表无值，t+1收益表有值：根据mask保留指标表值，原值为NaN，不影响计算，同样不参与计算
    2. 对于需要计算权重如rank方法：
        - 将mask后的指标表，转化为权重表
        - 再将权重表（`shift(1)`）与收益表相乘`DataFrame.mul`，`shift(1)`为了保证结果的日期是正确的
    3. 对于等权重计算如qcut方法
        - 则可以根据分组，直接计算组内均值。

    scheme = qcut 或 rank 或 zero
    - qcut
        - b_num，等证券数量分组：等权重
        - b_list，自定义分组：标记方法：[30,40,30]，等权重
        - 返回所有组数据
    - rank
        - 根据rank计算权重（做多rank最高，做多rank最低），并使得所有做多品种权重与所有做空品种权重，分别为1与-1
        - 注意使用sum时，即便行内所有值都为np.nan，求和结果为0，需要重新调整为np.nan
        - 返回一列结果
    - zero
        - 等权重
        - 排除carry为0，不升水也不贴水
        - 返回两列结果，升水与贴水

    Parameters
    ----------
    sig_raw : pd.DataFrame
        signal dataframe
    ret_raw : pd.DataFrame
        individual asset return dataframe
    scheme : str
        choose between: 'qcut', 'rank', 'zero'
    b_num : int, optional
        number of bins, each bin contains equal sample size, by default None
    b_list : list, optional
        list of to create unequal sample size bins,in format of `[30, 40, 30]`, by default None

    Returns
    -------
    result : pd.DataFrame
        time-series of all portfolios returns including long-short (high-minues-low) portfolio
    """

    # assertion sig_df, ret_df
    # 确保行列都一致，确保index和columns和表内容type
    sig_df, ret_df = sig_raw.copy(), ret_raw.copy()
    assert isinstance(sig_df, pd.DataFrame), 'sig_df应为DataFrame'
    assert isinstance(ret_df, pd.DataFrame), 'ret_df应为DataFrame'
    assert all(sig_df.columns == ret_df.columns), '列不一致'
    assert all(sig_df.index == ret_df.index), '行不一致'
    # 确保index为datetime，内容全为float
    assert isinstance(sig_df.index, pd.core.indexes.datetimes.DatetimeIndex) and isinstance(ret_df.index, pd.core.indexes.datetimes.DatetimeIndex), '两表的index为应为DatetimeIndex'
    assert all(sig_df.dtypes == float) and all(ret_df.dtypes == float), '两表内容应为float'
    # assertion scheme
    assert scheme in ['qcut', 'rank', 'zero'], 'scheme只能选择qcut或rank或zero三种计算方式'

    # 将sig_df向前一步，基于ret_df的时间戳（真实收益计算时间）
    # 根据下一期ret_df创建指标表mask，确保下一期ret存在
    # 若为下一期ret为np.nan，则标记为sig_fg对应位置np.nan，该个体不参与后续ret计算
    sig_df = sig_df.shift(1)[ret_df.notnull()]


    if scheme == 'qcut':
        # assertion
        assert (b_num is not None) ^ (b_list is not None), '提供b_num或b_list两者其一'
        if b_num:
            assert isinstance(b_num, int), 'b_num应为int'
        if b_list:
            assert isinstance(b_list, list), 'b_list应为list'
            assert sum(b_list) == 100, 'b_list之和应为100'

        # 统一使用bins_list输入
        if b_num is not None:
            bins_list = np.append([0], [i/b_num for i in range(1,b_num+1)])
        if b_list is not None:
            bins_list = np.append([0], np.cumsum(b_list).astype(float)/100)
        bins_num = len(bins_list)-1 # 分多少组
#         print(bins_list)

        # 若行全为np.nan则无法计算，进行分组，并标记为1,2,3,...(int)
        # rank_mask = sig_df[~sig_df.isna().all(axis=1)].apply(lambda x: pd.qcut(x,bins_list,range(1,bins_num+1)), axis=1)
        rank_mask = sig_df[~sig_df.isna().all(axis=1)].apply(lambda x: qcut_wrapper(x,bins_list,range(1,bins_num+1)), axis=1)
        # 输出结果
        result = pd.DataFrame()
        for group in range(1, bins_num+1):
            result[str(group)] = ret_df[rank_mask == group].mean(axis=1)
        result['H-L'] = result[str(bins_num)] - result['1']


    if scheme == 'rank':
        # 计算缩放量，使得总做多权重为1，总做空权重为-1
        # using lambda instead def function
        rank_scaler = lambda N: np.sum([i for i in [j - (N+1)/2 for j in range(1,N+1)] if i>0])
        # 先按大小，从小到大排序，rank越高数值越大
        wgt_df = sig_df.rank(axis=1)
        # 减去(n+1)/2，n为偶数一半为正一半为负，n为奇数中位数为零不进入计算
        wgt_df = wgt_df.apply(lambda x: (x-(x.count()+1)/2), axis=1)
        # scalar使得做多总权重为1，做空总权重为-1
        wgt_df = wgt_df.apply(lambda x: x/rank_scaler(x.count()), axis=1)
        # 输出结果
        result = pd.DataFrame(wgt_df.mul(ret_df).sum(axis=1), columns=['rank'])
        # 行内所有值为np.nan，sum结果为0，改为np.nan
        result[wgt_df.isna().all(axis=1)] = np.nan


    if scheme == 'zero':
        # 输出结果
        result = pd.DataFrame()
        result['<0'] = ret_df[sig_df<0].mean(axis=1)
        result['>0'] = ret_df[sig_df>0].mean(axis=1)
        result['H-L'] = result['>0'] - result['<0']

    return result