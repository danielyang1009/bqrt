
"""
Time-series Rgression
----------------------
"""

import numpy as np
import pandas as pd


# 其中包含''空的行
def show_ts_stat(df:pd.DataFrame, digi=4):

    result = pd.DataFrame(index=['***','**','*','sig','not_sig','total','avg','ttl_avg'], columns=df.columns, dtype=str)

    for col in df.columns:
        three_star = (df[col].apply(lambda x: x[-3:] == '***')).sum().sum()
        two_star_and_above = (df[col].apply(lambda x: x[-2:] == '**')).sum().sum()
        one_star_and_above = (df[col].apply(lambda x: x[-1:] == '*')).sum().sum()

        two_star = two_star_and_above - three_star
        one_star = one_star_and_above - two_star_and_above

        result.loc['***',col] = three_star
        result.loc['**', col] = two_star
        result.loc['*', col] = one_star
        result.loc['total',col] = df[col].replace('',np.nan).count()
        result.loc['sig',col] = one_star_and_above
        result.loc['not_sig',col] = result.loc['total',col] - one_star_and_above

    result['total'] = result.sum(axis='columns').astype(int).astype(str)

    float_df = strip_stars(df)
    mean = float_df.mean().round(digi).astype(str)
    result.loc['avg'] = mean
    result.loc[['avg','ttl_avg'],'total'] = ''
    result.loc['ttl_avg',result.columns[0]] = '{:.4f}'.format(round(float_df.stack().mean(),digi))

    return result.fillna('')


# 去除结果中的星，并转化为float格式
def strip_stars(df:pd.DataFrame):

    return df.applymap(lambda x: x.strip('*')).replace('',np.nan).astype('float')


# 可移除，功能放在common中的count star
def count_sig_col(data, var):

    assert isinstance(data, pd.DataFrame), 'data shall be pd.DataFrame'
    sig_count = sum([i=='*' for i in data[var].str[-1]])
    return sig_count


# 输入为singl_xvar_ts与multi_xvar_ts的结果（包含打星的系数和t值）
# 显示所有系数与显著程度，不显示第二行t值
def show_coe_only(s):

    return s.loc[s.index!='']


# 输入为singl_xvar_ts与multi_xvar_ts的结果（包含打星的系数和t值）
# 只显示显著的系数与显著程度，不显示第二行t值
def show_sig_only(s):

    if 'adj-r2' in s.columns:
        r2 = show_coe_only(s[['adj-r2']])
        s = show_coe_only(s).drop(columns=['adj-r2'])
        s = s[s.map(lambda x: x[-1]=='*')].fillna('')
        s = s.merge(r2,left_index=True,right_index=True,how='left')
    else:
        s = show_coe_only(s)
        s = s[s.map(lambda x: x[-1]=='*')].fillna('')
    return s


def singl_ts(yvar_data, yvar_list, xvar_data, xvar_list, controls=[], *, intercept=True, HAC=False, maxlags=None, digi=4, r2_pct=False):

    from statsmodels.api import OLS, add_constant

    if intercept:
        xvars = ['const']+xvar_list+controls
    else:
        xvars = xvar_list+controls

    s_cols = xvars+['adj-r2']
    s_rows = sum([[i,i+'_t'] for i in yvar_data[yvar_list]],[])
    s = pd.DataFrame(index=s_rows, columns=s_cols)

    for yvar in yvar_data[yvar_list]:
        Y = yvar_data[yvar]
        if intercept:
            X = add_constant(xvar_data[xvar_list+controls])
        else:
            X = xvar_data[xvar_list+controls]

        if HAC:
            # 用不用use_t没差别
            reg = OLS(Y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags':maxlags}, use_t=True)
        else:
            reg = OLS(Y, X, missing='drop').fit(use_t=True)

        for col in xvars:
            tval, pval = reg.tvalues[col], reg.pvalues[col]
            param = reg.params[col]
            stars = ''.join(['*' for i in [0.01, 0.05, 0.1] if pval<=i])
            # 【TODO】 考虑是否为stars加上padding
            # 后续需要用strip()剔除空格，但由于字体原因空格与数字所占空间未必能对齐
            # stars = stars.ljust(3,' ')
            s.loc[yvar,col] = '{:.{}f}'.format(param, digi) + stars
            s.loc[f'{yvar}_t', col] = f'({tval:.2f})' # 在excel会直接变为负数，需先将单元格设置为文本

        adj_r2 = reg.rsquared_adj
        # 是否放大为百分比r2
        if r2_pct == True:
            adj_r2 = adj_r2 * 100
        s.loc[yvar,'adj-r2'] = '{:.{}f}'.format(adj_r2, digi)

    s.index = sum([[i,''] for i in yvar_data[yvar_list]],[])
    return s.fillna('')


# 当xvar_list只有单变量时，即为单xvar检验，报告详细表所有系数，4位小数系数，并标记显著程度（打星），与adj-r方，下一行为括弧内（t值）
# 当xvar_list有多变量时，即为多xvar检验，仅报告xvar_list内变量，系数，4位小数，标注显著程度(打星)，下一行为括号内（t值）
def ts_reg(yvar_data, yvar_list, xvar_data, xvar_list, controls=[], *, intercept=True, HAC=False, maxlags=None, digi=4, r2_pct=False):

    assert isinstance(yvar_data, pd.DataFrame) and isinstance(xvar_data, pd.DataFrame), 'yvar_data and xvar_data shall be pd.DataFrame'
    assert all(yvar_data.index == xvar_data.index), 'yvar_data and xvar_data index not same'

    assert any([isinstance(yvar_list, list), isinstance(yvar_list, pd.Index)]), 'yvar_list shall be a list or pd.Index'
    assert all([y in yvar_data.columns for y in yvar_list]), 'Y variable(s) not in yvar_data'

    assert any([isinstance(xvar_list,list), isinstance(xvar_list, pd.Index)]), 'xvars shall be a list or pd.Index'
    assert all([x in xvar_data.columns for x in xvar_list]), 'X variable(s) not in xvar_data'

    if len(controls) != 0:
        assert isinstance(controls, list), 'control shall be a list'
        assert all([x in xvar_data.columns for x in controls]), 'control variable(s) not in xvar_data'
    if HAC:
        assert isinstance(maxlags,int), 'maxlags (int) is needed'

    if len(xvar_list) == 1:
        s = singl_ts(yvar_data, yvar_list, xvar_data, xvar_list, controls=controls, intercept=intercept, HAC=HAC, maxlags=maxlags, digi=digi, r2_pct=r2_pct)

    if len(xvar_list) > 1:
        s = []
        for xvar in xvar_list:
            xvar_reg = singl_ts(yvar_data, yvar_list, xvar_data, [xvar], controls=controls, intercept=intercept, HAC=HAC, maxlags=maxlags, digi=digi, r2_pct=r2_pct)
            s.append(xvar_reg[xvar].reset_index(drop=True))
        s = pd.DataFrame(s).T

    s.index = sum([[i,''] for i in yvar_data[yvar_list]],[])
    return s