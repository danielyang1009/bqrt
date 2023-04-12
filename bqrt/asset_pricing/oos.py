"""
Out-of-sample Predictability
----------------------------

- [ ] OOS R^2
- [ ] expanding/rolling window
"""

import numpy as np

def predicted_y(data, yvar, xvar_list, scheme, *, window):

    assert scheme in ['expanding','rolling'], 'scheme只能选择expanding和rolling两种方式'
    # window 为 expanding起始大小，或rolling滚动窗口大小

    import statsmodels.api as sm

    # 方便计算
    df = data.copy().reset_index()

    # 循环
    for end in range(window-1, len(df)):
        if scheme == 'expanding':
            start = 0
        if scheme == 'rolling':
            start = end - window + 1

        print(start, end)
        # 需要注意df[start:end]不包含end行，df.loc[start:end]根据index，包含end行
        # 剔除window中的最后一条作为prediction
        X_train = df.loc[start:end-1, xvar_list]
        y_train = df.loc[start:end-1, yvar]
#         print(df.loc[start:end])
#         print(X_train)
#         print(y_train)

        reg = sm.OLS(y_train, X_train, missing='drop').fit()
        X_test = df.loc[end, xvar_list]
#         print(X_test)
        y_pred = reg.predict(X_test)
        df.loc[end, 'y_pred'] = y_pred[0]
        df.loc[end, 'y_mean'] = df.loc[start:end-1, yvar].mean()

    df = df.set_index('index')
    df.index.name = None
    return df


def oos_r2(data, y, y_pred, y_bench):

    # na不会影响计算
    # df = data[[y, y_pred, y_bench]].dropna()

    ss_pred = np.sum((data[y] - data[y_pred])**2)
    ss_bench = np.sum((data[y] - data[y_bench])**2)
    print(ss_pred, ss_bench)

    oos_r2 = 1 - ss_pred / ss_bench
    # critifal value
    # oos_f = len(df) * ss_bench / ss_pred * oos_r2

    return oos_r2