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


def kwargs_test(**kwargs):
    """_summary_

    >>> kwargs_test(yvar='ret', xvar=['intercept','MKT'])
    1
    """

    if 'yvar' in kwargs and 'xvar' in kwargs:
        yvar = kwargs['yvar']
        xvar = kwargs['xvar']

    return yvar, xvar

def __np_ols(data, yvar, xvar, keep_r2=False):
    """
    Wrapper of `np.linalg.lstsq(a,b)`, which sovles `a @ x = b` for x.
    

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
    pd.Series
        least-squares solutions of x, if b is two-dimensional, then solutions in k columns
        
    Notes
    -----
    Under the hood, pseudoinverse is calculated using singular value decomposition (SVD), As any matrix can be decomposite as $A=U \Sigma V^T$, then pseudoinverse of matrix $A$ is $A^+ = V \Sigma^+ U^T$. `rcond` is used to set cut-off ratio for small singular values of in $\Sigma$. Setting `rcond=None` to silence the warning and use machine prcision as rcond parameter. 

    [What does the rcond parameter of numpy.linalg.pinv do?](https://stackoverflow.com/questions/53949202/what-does-the-rcond-parameter-of-numpy-linalg-pinv-do)

    [lstsq api](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
    """

    if keep_r2 == True:
        beta, ssr, _, _ = np.linalg.lstsq(data[xvar], data[yvar], rcond=None)
    
        if ssr.size == 0:
            X = np.mat(data[xvar])
            y = np.mat(data[yvar]).T
            y_hat = X @ np.mat(beta).T
            ssr = sum(np.square(y-y_hat)).item()
    
        # match statsmodel
        if any(i in ['intercept','const'] for i  in xvar):
            dof_model = len(xvar) - 1 # excluding intercept
            # dof_total = n - 1
            dof_total = data.shape[0] - 1
            # centered sst
            sst = sum((data[yvar] - data[yvar].mean())**2)
        else:
            dof_model = len(xvar)
            # dof_total = n
            dof_total = data.shape[0]
            # uncentered sst
            sst = sum((data[yvar]**2))

        dof_resid = dof_total - dof_model
        mse_total = sst / dof_total
        mse_resid = ssr / dof_resid
        r2 = 1 - ssr/sst
        adj_r2 = 1 - mse_resid / mse_total
        
        return pd.concat([pd.Series(beta), pd.Series(r2), pd.Series(adj_r2)])
    
    else:
        beta, _, _, _ = np.linalg.lstsq(data[xvar], data[yvar], rcond=None)

        return pd.Series(beta)


def __sm_ols(data, yvar, xvar, interp=False):

    import statsmodels.api as sm

    if interp:
        res = sm.OLS(data[yvar], sm.add_constant(data[xvar])).fit()
    else:
        res = sm.OLS(data[yvar], data[xvar]).fit()
    
    return res


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


def fm_constant_beta(data, entity, yvar, xvar):
    """
    Fisr step of Fama-Macbeth regression (unconditional)
    Time series regression for every i (asset) from all periods, getting unconditional (look-ahead bias) betas, `excess_return_t ~ lambda_t` same period matching regression.

    Parameters
    ----------
    data : _type_
        _description_
    entity : str
        column of test assets
    yvar : string
        excess return of test asset
    xvar : list of strings
        factor lambda

    Returns
    -------
    pd.DataFrame, index of i, shape of (data['i'].nunique(), len(xvar))
        return cross-sectional result of betas, list of estimated betas for every i

    Note
    ----
    Same as fama_macbeth groupby i instead of groupby t
    """

    estimated_betas = data.groupby(entity).apply(__np_ols, yvar, xvar)
    estimated_betas.columns = xvar

    return estimated_betas


def fm_rolling_beta(data, time, entity, yvar, xvar_list, window=120, min_nobs=None):
    
    from tqdm.notebook import tqdm
    from statsmodels.regression.rolling import RollingOLS
    
    if min_nobs is None:
        min_nobs = window
    
    # keep timestampe in rolling_betas
    data = data.set_index('date')
    rolling_beta = pd.DataFrame()
    for symb in tqdm(data[entity].unique()):
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
    fm_table = data[[entity, yvar]].reset_index()
    fm_table = fm_table.merge(rolling_beta, on=[time, entity], how='left')

    return fm_table[[time, entity, yvar] + xvar_list].sort_values([time, entity])


def fama_macbeth(data, time, yvar, xvar, keep_r2=False):
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
        estimated_lambdas = data.groupby(time).apply(__np_ols, yvar, xvar, keep_r2=True)
        # rename column names
        estimated_lambdas.columns = xvar + ['r2', 'adj-r2']
    else:
        estimated_lambdas = data.groupby(time).apply(__np_ols, yvar, xvar)
        # rename column names
        estimated_lambdas.columns = xvar

    return estimated_lambdas


def fm_summary(lambd, HAC=False, **kwargs):
    """
    Describe time-series of variables, typical for describing factor risk premium or lambdas, only describe a set of variables or result of single fama-macbeth regression

    Parameters
    ----------
    lambd : pd.DataFrame
        `time` variable as index, variables in columns, if `t` time periods with k variables then Dataframe has a shape of (t,k) 
    HAC : bool, optional
        using HAC estimator or not, need to specify `maxlags` if True, i.e `maxlags=8`

    Returns
    -------
    pd.DataFrame
        based on pandas describe function adding standard error, t-statistic and p-value
    """

    from scipy import stats
    import statsmodels.formula.api as smf

    s = lambd.describe().T
    # getting robust HAC estimators
    if HAC:
        if ('maxlags' in kwargs):
            maxlags = kwargs['maxlags']
            full_xvars = lambd.columns.to_list()
            std_error = []
            for var in full_xvars:
                # calculate individual Newey-West adjusted standard error using `smf.ols`
                reg = smf.ols('{} ~ 1'.format(var), data=lambd).fit(cov_type='HAC', cov_kwds={'maxlags':maxlags})
                std_error.append(reg.bse[0])
            s['std_error'] = std_error
        else:
            print('`maxlag` is needed to computer HAC')
    else:
        # nonrobust estimators
        s['std_error'] = s['std'] / np.sqrt(s['count'])

    # t-statistics
    s['tstat'] = s['mean'] / s['std_error']
    # 2-sided p-value for the t-statistic
    s['pval'] = stats.t.sf(np.abs(s['tstat']), s['count'] - 1) * 2

    return s


def fm_2nd_pass_reglist(data, time, reglist, HAC=False, **kwargs):
    """
    Running multiple only second-pass of fama-macbeth regression from `reglist`. When using firm characteristics as betas, there's no need to estimate betas from first pass with factor risk premium.  

    Parameters
    ----------
    data : pd.DataFrame
        dataframe contains excess return and factor beta (exposure) in long format (test asssts and t in rows, factors in columns)
    time : string
        column name of date/time/periods
    reglist : list
        list of R-like regression formula, last regression contains full x variables
    HAC : bool, optional
        using HAC estimator or not, need to specify `maxlags` if True, i.e `maxlags=8`

    Returns
    -------
    dictionary with keys of 'lambda' and 'summary'
        first regression results can be accessed from `summary['lambda'][0]` and `summary['summary'][0]` and so on
    """

    from tqdm.notebook import tqdm

    # initialized
    summary = {}
    summary['lambda'], summary['summary'] = [], []
    #  reglist for-loop
    for reg in tqdm(reglist, desc='Reg No.'):
        yvar, xvar_list = from_formula(reg)

        lambd = fama_macbeth(data, time, yvar, xvar_list)
        summary['lambda'].append(lambd)

        # HAC estimator
        if HAC:
            if ('maxlags' in kwargs):
                maxlags = kwargs['maxlags']
                s = fm_summary(lambd, HAC=True, maxlags=maxlags)
                summary['summary'].append(s)
            else:
                print('`maxlag` is needed to computer HAC')
        # nonrobust estimator
        else:
            s = fm_summary(lambd)
            summary['summary'].append(s)

    return summary


def fm_two_pass_reglist(data, time, entity, reglist, rolling:int=None, sc_interp=True):
    """
    Running multiple two-pass fama-macbeth regression from `reglist`, return time series of lambdas 

    standard fama-macbeth two-pass regression: 1) get estimated beta_{t} from `r_{t} ~ beta_{t} lambda_{t}` 2) get estimated lambda_{t+1} from `r_{t+1} ~ beta{t} \lambda{t+1}`

    [TODO] rolling beta
    
    Need to make sure last formula in `reglist` contains all x variables, which is determing maximum number of x variables.

    Parameters
    ----------
    data : pd.DataFrame
        dataframe contains excess return and factor beta (exposure) in long format (test asssts and t in rows, factors in columns)
    time : string
        column name of date/time/periods
    entity : str
        column of test assets
    reglist : list
        list of R-like regression formula, last regression contains full x variables
    sc_interp : bool, optional
        set an intercept or not for second-pass fama-macbeth regression

    Returns
    -------
    _type_
        _description_
    """

    from tqdm.notebook import tqdm

    # initialized
    summary = {}
    summary['beta'] = []
    summary['lambda'] = []
    summary['r2'] = []
    #  reglist for-loop
    for reg in tqdm(reglist, desc='Reg No.'):
        yvar, xvar_list = from_formula(reg)
        
        # first pass intercept by default
        data['intercept'] = 1

        # constant beta vs rolling beta
        if rolling != None:
            # TODO
            pass
        else:
            # index is symbol
            est_beta = fm_constant_beta(data, entity, yvar, ['intercept'] + xvar_list)
        summary['beta'].append(est_beta)

        # setting up for second pass, one-step lead-lag regression
        fm_table = data[[time,entity,yvar]].copy()
        fm_table.loc[:,yvar] = fm_table.groupby(entity)[yvar].shift(-1)
        fm_table = fm_table.dropna()
        fm_table = pd.merge(fm_table, est_beta, on=entity, how='left')

        # second pass interept, keep_r2=True
        if sc_interp:
            fm_table['intercept'] = 1
            res = fama_macbeth(fm_table, time, yvar, ['intercept'] + xvar_list, keep_r2=True)
            summary['lambda'].append(res[['intercept'] + xvar_list])
            summary['r2'].append(res[['r2', 'adj-r2']])
        else:
            res = fama_macbeth(fm_table, time, yvar, xvar_list, keep_r2=True)
            summary['lambda'].append(res[xvar_list])
            summary['r2'].append(res[['r2', 'adj-r2']])

    return summary


def fama_macbeth_summary(s, HAC=False, maxlags:int=None, params_digi:int=4, tstat_digi:int=2):
    """
    Taking results from `fm_two_pass_reglist` and `fm_2pass_reglist` returning financial research journal format summary table. Results are reported in either nonrobust or HAC estimators. 

    Parameters
    ----------
    s : dictionary
        dictionary of results from `fama_macbeth_reglist` funtion
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
    full_xvars = s['lambda'][-1].columns.to_list()

    # create reporting table
    summary_tbl_rows = sum([[var, '{}_t'.format(var)] for var in full_xvars], [])
    summary_tbl_cols = ['({})'.format(i+1) for i in range(len(s['lambda']))]
    summary_tbl = pd.DataFrame(index=summary_tbl_rows, columns=summary_tbl_cols)

    # putting params and t-value in place
    total_reg_no = len(s['lambda'])
    for reg_no in range(total_reg_no):
        model_no = '({})'.format(reg_no + 1)

        # HAC estimator
        if HAC:
            d = fm_summary(s['lambda'][reg_no], HAC=True, maxlags=maxlags)
        # nonrobust estimator
        else:
            d = fm_summary(s['lambda'][reg_no])
        
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
    summary_tbl.index = sum([[var, ''.format(var)] for var in full_xvars], [])

    # add r^2 and adj-R^2
    r2 = pd.DataFrame([i.mean() for i in s['r2']], index=['({})'.format(i) for i in range(1,total_reg_no +1)])
    r2 = r2.applymap(lambda x: '{:.{prec}f}'.format(x, prec=params_digi))
    summary_tbl = (pd.concat([summary_tbl.T, r2], axis=1)).T

    # replace NaN with whitespace for readability
    return summary_tbl.fillna('')