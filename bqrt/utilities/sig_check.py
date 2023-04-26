import pandas as pd


def check_ptable(data, *, threshold=None, star=False, columns=None, count=False):
    """
    Check p-value table

    Parameters
    ----------
    data : pd.DataFrame
        contains only p-values
    threshold : float, optional
        show only p-values below threshold, by default None
    star : bool, optional
        _description_, by default False
    columns : _type_, optional
        _description_, by default None
    count : float, optional
        count total pvalues < count, by default None

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

    sig_level = [.01, .05, .1]

    # 是否统计显著数目
    if count:
        for sig, stars in zip(sig_level[::-1], ['*','**','***']):
            ttl_count = data.count().sum()
            sig_count = data[data<=sig].count().sum()
            print('{:>3}: {}/{}'.format(stars, sig_count, ttl_count))

    ptable = data[columns].round(2)
    # 是否仅显示小于threshold
    if threshold != None:
        ptable = ptable[ptable <= threshold]

    # 是否打星
    if star == True:
        ptable = ptable.applymap(lambda x: str(x)+''.join(['*' for t in sig_level if x<=t]))

    return ptable.astype(str).replace({'nan':''})
