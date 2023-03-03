
"""
Time-series Rgression
----------------------
"""

import numpy as np
import pandas as pd


def sig_count(data, var):

    assert isinstance(data, pd.DataFrame), 'data shall be pd.DataFrame'

    sig_count = sum([i=='*' for i in data[var].str[-1]])

    return sig_count


# 单个xvar，显示截距，所有xvar（包括控制）的params和t，并打星
def singl_xvar_ts(yvar_data, yvar_list, xvar_data, xvar, controls=None, intercept=True, HAC=False, maxlags=None):

    assert isinstance(yvar_data, pd.DataFrame) and isinstance(xvar_data, pd.DataFrame), 'yvar_data and xvar_data shall be pd.DataFrame'
    assert any([isinstance(yvar_list, list), isinstance(yvar_list, pd.Index)]), 'yvar_list shall be a list or pd.Index'
    assert all([y in yvar_data.columns for y in yvar_list]), 'Y variable(s) not in yvar_data'
    assert isinstance(xvar, str), 'xvar_list shall be string'
    assert xvar in xvar_data.columns, 'X variable not in xvar_data'
    if controls is not None:
        assert isinstance(controls, list), 'control shall be a list'
        assert all([x in xvar_data.columns for x in controls]), 'control variable(s) not in xvar_data'
    assert all(yvar_data.index == xvar_data.index), 'yvar_data and xvar_data index not same'
    if HAC:
        assert isinstance(maxlags,int), 'maxlags (int) is needed'

    from statsmodels.api import OLS, add_constant

    s_cols = ['const']+[xvar]+controls if intercept else [xvar]+controls
    s_rows = sum([[i,i+'_t'] for i in yvar_data.columns],[])
    s = pd.DataFrame(index=s_rows, columns=s_cols)

    for yvar in yvar_data.columns:
        Y = yvar_data[yvar].copy()
        if intercept:
            X = add_constant(xvar_data[[xvar]+controls]).copy()
        else:
            X = xvar_data[[xvar]+controls].copy()

        if HAC:
            # 用不用use_t没差别
            reg = OLS(Y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags':maxlags})
        else:
            reg = OLS(Y, X, missing='drop').fit(use_t=True)

        for col in s_cols:
            tval, pval = reg.tvalues[col], reg.pvalues[col]
            param = reg.params[col]
            stars = ''.join(['*' for i in [0.01, 0.05, 0.1] if pval<=i])
            s.loc[yvar,col] = f'{param:.4f}' + stars
            s.loc[f'{yvar}_t', col] = f'({tval:.2f})'

    s.index = sum([[i,''] for i in yvar_data.columns],[])
    return s


# 多个xvar，仅显示所有xvar的params和t，并打星
def multi_xvar_ts(yvar_data, yvar_list, xvar_data, xvar_list, controls=None, intercept=True, HAC=False, maxlags=None):

    assert any([isinstance(xvar_list, list), isinstance(xvar_list, pd.Index)]), 'xvar_list shall be a list or pd.Index'

    s = []
    for xvar in xvar_list:
        xvar_reg = singl_xvar_ts(yvar_data, yvar_list, xvar_data, xvar, controls=controls, intercept=intercept, HAC=HAC, maxlags=maxlags)
        s.append(xvar_reg[xvar].reset_index(drop=True))

    s = pd.DataFrame(s).T
    s.index = sum([[i,''] for i in yvar_data.columns],[])

    return s


# 所有ts回归结果取均值
def standard_st():
    pass