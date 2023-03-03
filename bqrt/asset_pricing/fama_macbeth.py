"""
Fama-Macbeth Rgression
----------------------

Run single or multiple (from `reglist`) fama-macbeth regression and print results. Results are organized in conventional financial research journal format with different models in columns and factors in rows, also mark with asterisks (stars) for significant level.
"""

import numpy as np
import pandas as pd


def print_test():
    """_summary_

    >>> 1
    1
    >>> 2
    2
    """

    print(3)


def hac_maxlags(t):
    """
    Calculate `maxlags` for Heteroskedasticity and Autocorrelation Consistent (HAC) estimator or Newey–West estimator

    ..math::

        J = [4 \\times T/100 ]^{2/9}

    Parameters
    ----------
    t : int
        length of time series

    Returns
    -------
    J : float
        maxlags

    Reference
    ---------
    [1] Newey, Whitney K., and Kenneth D. West, 1987, A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix, Econometrica 55, 703–708.

    """

    return int(np.floor(4*(t/100)**(2/9)))


def from_formula(reg_formula: str):
    """
    Breaking list of regression formula into y and x variables

    Parameters
    ----------
    reg_formula : string
        string of regression formula

    Returns
    -------
    (yvar, xvar_list) : tuple

    yvar : string
        string of y variable name
    xvar_list : list
        list of x variable names

    Example
    -------
    >>> from_formula('y ~ intercept + x1')
    ('y', ['intercept', 'x1'])
    >>> from_formula('ret~1+x1+x2')
    ('ret', ['1', 'x1', 'x2', 'x3'])
    """

    yvar = reg_formula.split('~')[0].strip()
    x_string = reg_formula.split('~')[1].strip()
    xvar_list = [x.strip() for x in x_string.split('+')]

    return yvar, xvar_list


def np_ols(data, yvar, xvar_list, keep_r2=False):
    """
    Wrapper of `np.linalg.lstsq(a,b)`, which sovles `a @ x = b` for x. Also calculate r2 and adj-r2

    Parameters
    ----------
    data : pd.DataFrame
        data
    yvar : string
        string of y variable
    xvar : list of string
        list of string of x variables
    keep_r2 : bool
        whether keep r-squared of adj. r-squared in result

    Returns
    -------
    pd.DataFrame
        Least-squares solutions of x, if b is k dimensional, then solutions in k columns. if `keep_r2 == True`, add two columns of `r2` and `adj_r2`

    Notes
    -----
    Under the hood, pseudoinverse is calculated using singular value decomposition (SVD), As any matrix can be decomposite as $A=U \Sigma V^T$, then pseudoinverse of matrix $A$ is $A^+ = V \Sigma^+ U^T$. `rcond` is used to set cut-off ratio for small singular values of in $\Sigma$. Setting `rcond=None` to silence the warning and use machine prcision as rcond parameter.

    [What does the rcond parameter of numpy.linalg.pinv do?](https://stackoverflow.com/questions/53949202/what-does-the-rcond-parameter-of-numpy-linalg-pinv-do)

    [lstsq api](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
    """

    if keep_r2 == True:
        beta, ssr, _, _ = np.linalg.lstsq(data[xvar_list], data[yvar], rcond=None)

        if ssr.size == 0:
            X = np.mat(data[xvar_list])
            y = np.mat(data[yvar]).T
            y_hat = X @ np.mat(beta).T
            ssr = sum(np.square(y-y_hat)).item()

        # match statsmodel
        if any(i in ['intercept','Intercept','const'] for i  in xvar_list):
            dof_model = len(xvar_list) - 1 # excluding intercept
            # dof_total = n - 1
            dof_total = data.shape[0] - 1
            # centered sst
            sst = sum((data[yvar] - data[yvar].mean())**2)
        else:
            dof_model = len(xvar_list)
            # dof_total = n
            dof_total = data.shape[0]
            # uncentered sst
            sst = sum((data[yvar]**2))

        # calcuate r2 / adj_r2
        dof_resid = dof_total - dof_model
        mse_total = sst / dof_total
        mse_resid = ssr / dof_resid
        r2 = 1 - ssr/sst
        adj_r2 = 1 - mse_resid / mse_total

        return pd.concat([pd.Series(beta), pd.Series(r2), pd.Series(adj_r2)])

    else:
        beta, _, _, _ = np.linalg.lstsq(data[xvar_list], data[yvar], rcond=None)

        return pd.Series(beta)


def scipy_ols(data, yvar, xvar_list):

    from scipy.linalg import lstsq

    beta, _, _, _ = lstsq(data[xvar_list], data[yvar], rcond=None)

    return pd.series(beta)


def sm_ols(data, yvar, xvar_list):

    import statsmodels.api as sm

    res = sm.OLS(data[yvar], data[xvar_list]).fit(params_only=True)

    return res.params.reset_index(drop=True)


def fm_constant_beta(data, yvar, xvar_list, time='date', entity='symbol'):
    """
    First pass of Fama-Macbeth regression (unconditional)
    Time series regression (`excess_return_t ~ factor_t` same period matching regression.) for every i (asset) from all periods, getting unconditional (look-ahead bias) betas. Getting ready `fp_table` for second pass regression.

    Parameters
    ----------
    data : pd.DataFrame
        long format dataframe for every date, every test asset, excess return and factor returns
    yvar : string
        column name of test asset excess return
    xvar_list : list of strings
        list of x variable column names
    time: str
        column name of datetime, by default 'date'
    entity : str
        column name of test asset symbols, by default 'symbol'

    Returns
    -------
    fp_table: pd.DataFrame
        long format dataframe for second pass regression

    Note
    ----
    First pass groupby time, second pass groupby entity
    """

    constant_beta = data.groupby(entity).apply(np_ols, yvar, xvar_list)
    constant_beta.columns = xvar_list
    constant_beta = constant_beta.reset_index()

    # no need to shift, since all betas are same across time
    fp_table = data[[time, entity, yvar]].copy()
    fp_table = pd.merge(fp_table, constant_beta, on=entity, how='left')

    return fp_table.sort_values([time, entity])


def fm_rolling_beta(data, yvar, xvar_list, time='date', entity='symbol', window=120, min_nobs=None):
    """
    First pass regression of Fama-Macbeth regression (conditional)
    Times series regression (`excess_return_t ~ factor_t` same period matching regression) of rolling windows, calcuate rolling betas without look-ahead bias

    Parameters
    ----------
    data : pd.DataFrame
        long format dataframe for every date, every test asset, excess return and factor returns
    yvar : string
        column name of test asset excess return
    xvar_list : list of strings
        list of x variable column names
    time: str
        column name of datetime, by default 'date'
    entity : str
        column name of test asset symbols, by default 'symbol'
    window : int, optional
        rolling window, by default 120
    min_nobs : _type_, optional
        minimum number of observation for rolling regression, by default None

    Returns
    -------
    fp_table: pd.DataFrame
        long format dataframe for second pass regression

    Note
    ----
    First pass groupby time, second pass groupby entity
    """

    from tqdm.notebook import tqdm
    from statsmodels.regression.rolling import RollingOLS

    if min_nobs is None:
        min_nobs = window

    data = data.set_index('date')
    rolling_beta = pd.DataFrame()
    # run rolling regressions for every test aseets
    for symb in tqdm(data[entity].unique(), leave=False):
        Y = data[data[entity] == symb][yvar]
        X = data[data[entity] == symb][xvar_list]
        rolling_ols = RollingOLS(Y, X, window=window, min_nobs=min_nobs)
        to_add = rolling_ols.fit(params_only=True).params[xvar_list]
        to_add[entity] = symb
        rolling_beta = pd.concat([rolling_beta, to_add],axis='rows')

    # setup for second step lead-lag regression
    # must have time, entity as index, or group columns will be drop after groupby().shift()
    rolling_beta = rolling_beta.set_index([rolling_beta.index, entity])
    rolling_beta = rolling_beta.groupby(entity).shift(1).reset_index()
    fp_table = data[[entity, yvar]].reset_index()
    fp_table = fp_table.merge(rolling_beta, on=[time, entity], how='left')

    return fp_table.sort_values([time, entity])


def fama_macbeth(data, yvar, xvar_list, time='date', keep_r2=False):
    """
    Fama-macbeth regression (cross-sectional) for every t, `excess_return ~ beta' lambda`, regressing ret on beta to get time series of lambdas (factor risk premium)

    Parameters
    ----------
    data : pd.DataFrame
        dataframe contains excess return and factor beta (exposure) can be firm characteristics or estimated betas
    time : string
        column of date/time/periods
    yvar : string
        excess return of test asset
    xvar : list of strings
        factor betas
    keep_r2 : bool
        whether keep r-squared of adj. r-squared in result

    Returns
    -------
    pd.DataFrame, time as index, shape of (time, len(xvar))
        return time series result of estimated lambdas (factor risk premium), list of estimated lambdas for every t

    Notes
    -----
    If intercept is needed, add to xvar list.

    Reference
    ---------
    [1] Fama, Eugene F., and James D. MacBeth, 1973, Risk, Return, and Equilibrium: Empirical Tests, Journal of Political Economy 81, 607–636.

    """

    if keep_r2:
        estimated_lambdas = data.groupby(time).apply(np_ols, yvar, xvar_list, keep_r2=True)
        # rename column names
        estimated_lambdas.columns = xvar_list + ['r2', 'adj-r2']
    else:
        estimated_lambdas = data.groupby(time).apply(np_ols, yvar, xvar_list)
        # rename column names
        estimated_lambdas.columns = xvar_list

    return estimated_lambdas


def fm_2nd_pass_reglist(data, reglist, time='date', interp=True):
    """
    Running multiple only second-pass of fama-macbeth regression from `reglist`. When using firm characteristics as betas, there's no need to estimate betas from first pass with factor risk premium.

    Parameters
    ----------
    data : pd.DataFrame
        dataframe contains excess return and factor beta (exposure) in long format (test asssts and t in rows, factors in columns)
    reglist : list
        list of R-like regression formula, last regression contains full x variables
    time: str
        column name of datetime, by default 'date'

    Returns
    -------
    dictionary with keys of 'lambda' and 'summary'
        first regression results can be accessed from `summary['lambda'][0]` and `summary['summary'][0]` and so on
    """

    from tqdm.notebook import tqdm

    # initialized
    summary = {}
    summary['lambda'] = []
    summary['r2'] = []

    for reg in tqdm(reglist, desc='Reg No.'):
        yvar, xvar_list = from_formula(reg)

        if interp:
            data['intercept'] = 1
            res = fama_macbeth(data, yvar, ['intercept'] + xvar_list, time=time, keep_r2=True)
            summary['lambda'].append(res[['intercept']+ xvar_list])
            summary['r2'].append(res[['r2', 'adj-r2']])
        else:
            res = fama_macbeth(data, yvar, xvar_list, time=time, keep_r2=True)
            summary['lambda'].append(res[xvar_list])
            summary['r2'].append(res[['r2', 'adj-r2']])

    return summary


def fm_two_pass_reglist(data, reglist, time='date', entity='symbol', window=None, min_nobs=None, sp_interp=True):
    """
    Running multiple two-pass fama-macbeth regression from `reglist`, return time series of lambdas

    standard fama-macbeth two-pass regression: 1) get estimated beta_{t} from `r_{t} ~ beta_{t} lambda_{t}` 2) get estimated lambda_{t+1} from `r_{t+1} ~ beta{t} \lambda{t+1}`

    Need to make sure last formula in `reglist` contains all x variables, which is determing maximum number of x variables.

    Parameters
    ----------
    data : pd.DataFrame
        dataframe contains excess return and factor beta (exposure) in long format (test asssts and t in rows, factors in columns)
    reglist : list
        list of R-like regression formula, last regression contains full x variables
    sp_interp : bool, optional
        set an intercept or not for second-pass fama-macbeth regression
    time: str
        column name of datetime, by default 'date'
    entity : str
        column name of test asset symbols, by default 'symbol'

    Returns
    -------
    summary: dict
        with keys of 'fp_table', 'lambda', 'r2', accessing `fp_table` result from first item of reglist by `summary['fp_table][0]`
    """

    from tqdm.notebook import tqdm

    assert isinstance(reglist, list), 'reglist must be a list'

    # initialized
    summary = {}
    summary['fp_table'] = [] # first-pass result
    summary['lambda'] = [] # second-pass
    summary['r2'] = [] # second_pass

    data = data.sort_values([time, entity]).reset_index()
    #  reglist for-loop
    for reg in tqdm(reglist, desc='Reg No.'):
        yvar, xvar_list = from_formula(reg)

        # first pass
        # intercept by default
        # comment out these two lines to regress without intercept
        data['intercept'] = 1
        xvar_list = ['intercept'] + xvar_list

        # constant beta vs rolling beta
        if window is not None:
            fp_table = fm_rolling_beta(data, yvar, xvar_list, time=time, entity=entity, window=window, min_nobs=min_nobs)
        else:
            fp_table = fm_constant_beta(data, yvar, xvar_list, time=time, entity=entity)
        summary['fp_table'].append(fp_table)

        # second pass
        # nan create 'SVD did not converge in Linear Least Squares' error
        fp_table = fp_table.dropna().copy()
        if sp_interp:
            fp_table['intercept'] = 1
            res = fama_macbeth(fp_table, yvar, xvar_list, time=time, keep_r2=True)
            summary['lambda'].append(res[xvar_list])
            summary['r2'].append(res[['r2', 'adj-r2']])
        else:
            # remove intercept from xvar_list
            if 'intercept' in xvar_list:
                xvar_list.remove('intercept')
            res = fama_macbeth(fp_table, yvar, xvar_list, time=time, keep_r2=True)
            summary['lambda'].append(res[xvar_list])
            summary['r2'].append(res[['r2', 'adj-r2']])

    return summary


def desc_lambda(lambd, HAC:bool=False, maxlags:int=None):
    """
    [TODO] one-side t-test

    Describe time-series of variables, typical for describing factor risk premium or lambdas, only describe a set of variables or result of single fama-macbeth regression

    Parameters
    ----------
    lambd : pd.DataFrame
        datetime index, factor lambda in columns, if `t` time periods with k variables then Dataframe has a shape of (t,k)
    HAC : bool, optional
        using HAC estimator or not, need to specify `maxlags` if True, i.e `maxlags=8`

    Returns
    -------
    pd.DataFrame
        based on pandas describe function adding standard error, t-statistic and p-value
    """

    from scipy import stats
    import statsmodels.formula.api as smf

    if HAC:
        assert isinstance(maxlags, int), '`maxlags` (int) is needed'

    s = lambd.describe().T
    # getting robust HAC estimators of std_error
    if HAC:
        xvar_list = lambd.columns.to_list()
        std_error = []
        for var in xvar_list:
            # calculate individual Newey-West adjusted standard error using `smf.ols`
            reg = smf.ols('{} ~ 1'.format(var), data=lambd).fit(cov_type='HAC', cov_kwds={'maxlags':maxlags}, use_t=True)
            std_error.append(reg.bse[0])
        s['std_error'] = std_error
    # nonrobust estimators
    else:
        s['std_error'] = s['std'] / np.sqrt(s['count'])

    # t-statistics
    s['tstat'] = s['mean'] / s['std_error']
    # 2-sided p-value for the t-statistic
    s['pval'] = stats.t.sf(np.abs(s['tstat']), s['count'] - 1) * 2

    return s


def fm_summary(s, HAC=False, maxlags:int=None, params_digi:int=4, tstat_digi:int=2):
    """
    Taking results from `fm_2nd_pass_reglist` and `fm_two_pass_reglist` returning financial research journal format summary table. Results can be reported in either nonrobust or HAC estimators.

    Parameters
    ----------
    s : dictionary
        dictionary of results from `fm_2nd_pass_reglis` or `fm_two_pass_reglist` funtion
    HAC : bool, optional
        using HAC estimator or not, need to specify `maxlags` if True, i.e `maxlags=8`
    maxlags : int, optional
        maximum lags for HAC estimator
    params_format : str, optional
        number of digits to keep for params (coefficient), in this case mean of time series lambdas, by default '{:.4f}'
    tstat_format : str, optional
        number of digits to keep for t-statistics, adding parentheses '(x.xx)' format for better readability, by default '({:.2f})'

    Returns
    -------
    pd.DataFrame
        summary table of mean of times series lambdas (with significant level star mark) and t statistics in paratheses
    """

    # using last formula to get all x variables
    xvar_list = s['lambda'][-1].columns.to_list()

    # create reporting table
    summary_tbl_rows = sum([[var, '{}_t'.format(var)] for var in xvar_list], [])
    summary_tbl_cols = ['({})'.format(i+1) for i in range(len(s['lambda']))]
    summary_tbl = pd.DataFrame(index=summary_tbl_rows, columns=summary_tbl_cols)

    # putting params and t-value in place
    total_reg_no = len(s['lambda'])
    for reg_no in range(total_reg_no):
        model_no = '({})'.format(reg_no + 1)

        # HAC estimator
        if HAC:
            d = desc_lambda(s['lambda'][reg_no], HAC=True, maxlags=maxlags)
        # nonrobust estimator
        else:
            d = desc_lambda(s['lambda'][reg_no])

        for var in d.index.to_list():
            # getting p-values to determine significant level
            pval = d.loc[var, 'pval']
            param = '{:.{prec}f}'.format(d.loc[var, 'mean'], prec=params_digi)
            # add significant level star mark to params
            if (pval <= 0.1) & (pval > 0.05):
                param = param + '*'
            elif (pval <= 0.05) & (pval > 0.01):
                param = param + '**'
            elif pval <= 0.01:
                param = param + '***'

            # filling param
            summary_tbl.loc[var, model_no] = param
            # filling t-statistics
            summary_tbl.loc['{}_t'.format(var), model_no] = '({:.{prec}f})'.format(d.loc[var, 'tstat'], prec=tstat_digi)

    # replacing `var_t` index with whitespace,
    summary_tbl.index = sum([[var, ''.format(var)] for var in xvar_list], [])

    # add r^2 and adj-R^2
    r2 = pd.DataFrame([i.mean() for i in s['r2']], index=['({})'.format(i) for i in range(1,total_reg_no +1)])
    r2 = r2.applymap(lambda x: '{:.{prec}f}'.format(x, prec=params_digi))
    summary_tbl = (pd.concat([summary_tbl.T, r2], axis=1)).T

    # replace NaN with whitespace for readability
    return summary_tbl.fillna('')