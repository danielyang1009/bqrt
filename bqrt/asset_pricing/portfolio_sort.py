"""
Portfolio Sort
--------------

TODO: 
- 增加min_nobs，数量太少无法分组
- 增加计算ret的权重value-weighted（同期）
- 考虑双重排序，先对市值排序，而后对beta或sig_df排序
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
    ret_raw : _type_
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
    assert all(mv_df.columns == ret_df.columns), '列不一致'
    assert all(mv_df.index == ret_df.index), '行不一致'
    
    # 计算每天总市值
    ttl_mv_by_row = mv_df.sum(axis='columns')
    # 计算权重 = 市值/总市值
    mv_df = mv_df.div(ttl_mv_by_row, axis='rows')
    return mv_df.mul(ret_df).sum(axis='columns')


def portfolio_sort(sig_raw, ret_raw, scheme, b_num=None, b_list=None):
    """
    Calculating 1 step forward portfolios return based on signal dataframe, by different scheme methods. All inputs are in format of wide-table (date as index, individual asset as columns).

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

#     print(b_num, b_list)
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
        rank_mask = sig_df[~sig_df.isna().all(axis=1)].apply(lambda x: pd.qcut(x,bins_list,range(1,bins_num+1)), axis=1)
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


# archive
def port_sort(data, time, entity, ret, criteria, num=10, weight=None, min_entities=400):
    
    # will alter data, set copy
    df = data.copy()
    
#     forward criteria
    df[criteria] = df.groupby(entity)[criteria].shift(1)
    # lag return
    # df[ret] = df.groupby(entity)[ret].shift(-1)
    df = df.dropna().reset_index(drop=True)
    
    # checking min entities
    size_list = df.groupby(time).size().to_frame().reset_index()
    size_list.columns = [time,'size']
    print('Skip days with less {} entities'.format(min_entities))
    print(size_list[size_list['size'] < min_entities][time])
    time_filter = size_list[size_list['size'] > min_entities][time].sort_values()
    
    # setup dataframe, date as index
    result = pd.DataFrame(np.zeros((len(time_filter),num)))
    result.index = time_filter
    result.columns = np.arange(1,num+1)
    
    for date in tqdm(result.index):
        ref = df[df[time] == date].copy()

        # accending 10 -> biggest criteria group
        ref.loc[:,'rank'] = pd.qcut(ref[criteria], num, labels=np.arange(1,num+1))
        for rank in np.arange(1,num+1):
            if weight == None:
                result.loc[result.index == date, rank] = ref[ref['rank'] == rank][ret].mean()
            else:
                result.loc[result.index == date, rank] = ref[ref['rank'] == rank][ret] * ref[ref['rank'] == rank][criteria] / ref[ref['rank'] == rank][criteria].sum()
    
    result.columns = [str(i) for i in np.arange(1,num+1)]
    result['{}-1'.format(num)] = result[str(num)] - result['1']
    return result

    
def port_plot(pa_result, skip=True):
    if skip:
        (pa_result.drop(columns=['10-1'])+1).cumprod().plot(figsize=(12,6),grid=True)
    else:
        (pa_result+1).cumprod().plot(figsize=(12,6),grid=True)