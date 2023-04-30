import pandas as pd
import numpy as np

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
        _description_, by default None
    count : float, optional
        count total significant level, by default False
    digi : int, optional
        round to decimal
    Returns
    -------
    ptable : pd.DataFrame
        return result, if none of the optional arguement is set, then return original ptable with 2 decimal
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
        ptable = ptable.applymap(lambda x: str(np.round(x,digi))+''.join(['*' for t in [.01, .05, .1] if x<=t]))

    if digi != None:
        ptable = ptable.round(digi)

    # 剔除nan，保持原格式输出
    ptable = ptable.astype(str).replace({'nan':''})
    result = data.copy()
    for col in columns:
        result[col] = ptable[col]
    return result
