
"""
Panel Rgression
---------------

Using linearmodels to perform fixed effect panel regression.
"""
import pandas as pd


def panel_summary(data, reglist, entity='symbol', time='date', lead_lag=True, intercept=True, coef_digi=2, r2_digi=3, r2_pct=True, annual_factor=1):

    from tqdm.notebook import tqdm
    import statsmodels.api as sm
    from linearmodels.panel import PanelOLS
    from .time_series import from_formula

    # 利用最后一行reglist包含所有自变量
    xvar_list = from_formula(reglist[-1])[1]
    # print(xvar_list)
    if intercept:
        # 所有自变量包含常数
        xvar_list = ['const'] + xvar_list

    # 创建原始报表
    info_rows = ['Entity FE','Time FE', 'Obs.', 'R^2', 'Adj R^2']
    s_rows = sum([[var, '{}_t'.format(var)] for var in xvar_list], []) + info_rows
    s_cols = ['({})'.format(i+1) for i in range(len(reglist)*3)] # 包含三种固定效应模型
    s = pd.DataFrame(index=s_rows, columns=s_cols)

    col_count = 1
    for reg in tqdm(reglist,'Model No.'):
        yvar, xvars = from_formula(reg)

        # 设置格式进行面板回归
        panel_df = data.set_index([entity,time])[[yvar] + xvars]

        if lead_lag:
            panel_df[xvars] = panel_df.groupby(entity)[xvars].shift(1)

        panel_df = panel_df.dropna() # 之前合并数据也会产生NA，lead-lag会产生NA，剔除NA

        # 设置变量
        Y = panel_df[yvar] * annual_factor
        X = panel_df[xvars] * annual_factor
        if intercept:
            X = sm.add_constant(X) # 控制常数项

        # 全部三种固定效果
        for i in range(1,4):
            # print(xvars, i)
            if i == 1:
                model = PanelOLS(Y, X, entity_effects=True)
            if i == 2:
                model = PanelOLS(Y, X, time_effects=True)
            if i == 3:
                model = PanelOLS(Y, X, entity_effects=True, time_effects=True)

            # 拟合模型
            reg = model.fit()
            for var in ['const'] + xvars if intercept else xvars:
                tval, pval = reg.tstats[var], reg.pvalues[var]
                # print(tval,pval)
                param = reg.params[var]
                stars = ''.join(['*' for i in [0.01, 0.05, 0.1] if pval<=i]) # 打星
                col_name = '({})'.format(col_count) # 列名称
                s.loc[var, col_name] = '{:.{}f}'.format(param, coef_digi) + stars
                s.loc[f'{var}_t', col_name] = f'({tval:.2f})' # 在excel会直接变为负数，需先将单元格设置为文本
                # 固定效应比较
                if i == 1:
                    s.loc['Entity FE',col_name] = 'X'
                if i == 2:
                    s.loc['Time FE',col_name] = 'X'
                if i == 3:
                    s.loc['Entity FE',col_name] = 'X'
                    s.loc['Time FE',col_name] = 'X'

                s.loc['Obs.',col_name] = reg.nobs # 样本数
                # R方
                n = reg.nobs
                k = len(xvars)
                r2 = reg.rsquared
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
                if r2_pct:
                    r2 *= 100
                    adj_r2 *= 100
                s.loc['R^2',col_name] = '{:.{}f}'.format(r2, r2_digi)
                s.loc['Adj R^2',col_name] = '{:.{}f}'.format(adj_r2, r2_digi)

            col_count+=1

    s.index = sum([[i,''] for i in xvar_list],[]) + info_rows
    return s.fillna('')