
"""
Fama-Macbeth Rgression
----------------------

Run single or multiple (from `reglist`) fama-macbeth regression and print results. Results are organized in conventional financial research journal format with different models in columns and factors in rows, also mark with asterisks (stars) for significant level.
"""

import numpy as np
import pandas as pd


def print_test(x):
    """_summary_

    >>> 1
    1
    >>> 2
    2
    """

    print(4)


def __np_ols(data, yvar, xvar):
    """
    Wrapper of np.linalg.lstsq

    Notes
    -----
    Under the hood, pseudoinverse is calculated using singular value decomposition (SVD), As any matrix can be decomposite as $A=U \Sigma V^T$, then pseudoinverse of matrix $A$ is $A^+ = V \Sigma^+ U^T$. `rcond` is used to set cut-off ratio for small singular values of in $\Sigma$. Setting `rcond=None` to silence the warning and use machine prcision as rcond parameter. 

    [What does the rcond parameter of numpy.linalg.pinv do?](https://stackoverflow.com/questions/53949202/what-does-the-rcond-parameter-of-numpy-linalg-pinv-do)

    """

    betas,_,_,_ = np.linalg.lstsq(data[xvar], data[yvar], rcond=None)

    return pd.Series(betas.flatten())


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

    return np.floor(4*(t/100)**(2/9))


def from_formula(reg_formula:str):
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
    xvar_list = [ x.strip() for x in x_string.split('+') ]

    return yvar, xvar_list


def fama_macbeth(data, t, yvar, xvar):
    """
    Fama-macbeth regression for every t, `excess return ~ beta' lambda`, regressing ret on beta to get time series of lambdas (factor risk premium)

    Parameters
    ----------
    data : pd.DataFrame
        dataframe contains excess return and factor beta (exposure)
    t : string
        column name of date/time/periods
    yvar : string
        excess return of test asset
    xvar : list of strings
        factor betas

    Returns
    -------
    pd.DataFrame, shape of (len(t),len(xvar))
        return time series result of lambdas (factor risk premium)
    
    Notes
    -----
    If intercept is needed, add to xvar list.

    Reference
    ---------
    [1] Fama, Eugene F., and James D. MacBeth, 1973, Risk, Return, and Equilibrium: Empirical Tests, Journal of Political Economy 81, 607–636.

    """

    # running cross-sectional ols for every t, get time series lambdas
    d = (data.groupby(t).apply(__np_ols, yvar, xvar))
    # rename column names
    d.columns = xvar

    return d


def get_summary(lambd, HAC=False, **kwargs):
    """_summary_

    Parameters
    ----------
    lambd : _type_
        _description_
    HAC : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """

    import statsmodels.formula.api as smf
    from scipy import stats
    
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
    s['pval'] = stats.t.sf(np.abs(s['tstat']), s['count']-1) * 2 

    return s


def fama_macbeth_reglist(data, t, reglist, HAC=False, **kwargs):
    """
    Running multiple fama-macbeth regression from `reglist`, return time series of lambdas and their descriptive statistics as well as t-statistics, stantadard errors (nonrobust or HAC) and p-values.
    
    Need to make sure last formula in `reglist` contains all x variables, which is determing maximum number of x variables.

    Parameters
    ----------
    data : pd.DataFrame
        dataframe contains excess return and factor beta (exposure) in long format (test asssts and t in rows, factors in columns)
    t : string
        column name of date/time/periods
    reglist : list
        list of R-like regression formula
    HAC : bool, optional
        using HAC estimator or not, need to specify `maxlags` if True

    Returns
    -------
    dictionary with keys of 'lambda' and 'summary'
        first regression results can be accessed from `summary['lambda'][0]` and `summary['summary'][0]` and so on
    """

    # initialized
    summary = {}
    summary['lambda'], summary['summary'] = [], []
    #  reglist for-loop
    for reg in reglist:
        yvar, xvar_list = from_formula(reg)
        lambd = fama_macbeth(data, t, yvar, xvar_list)
        summary['lambda'].append(lambd)

        # HAC estimator
        if HAC:
            if ('maxlags' in kwargs):
                maxlags = kwargs['maxlags']
                s = get_summary(lambd, HAC=True, maxlags=maxlags)
                summary['summary'].append(s)
            else:
                print('`maxlag` is needed to computer HAC')
        # nonrobust estimator
        else:
            s = get_summary(lambd)
            summary['summary'].append(s)

    return summary


def fama_macbeth_summary(s, params_format='{:.4f}', tstat_format='({:.2f})'):
    """
    Running multiple fama-macbeth regression from `reglist`, returning financial research journal format summary table.

    Parameters
    ----------
    s : dictionary
        dictionary of results from `fama_macbeth_reglist` funtion
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
    full_xvars = s['summary'][-1].index.to_list()
    # create DataFrame
    summary_table_rows = sum([[var, '{}_t'.format(var)] for var in full_xvars], [])
    summary_table_cols = ['({})'.format(i+1) for i in range(len(s['summary']))]
    summary_table = pd.DataFrame(index=summary_table_rows, columns=summary_table_cols)
    
    # putting params and t-value in place
    for reg_no in range(len(s['summary'])):
        model_no = '({})'.format(reg_no+1)
        for var in s['summary'][reg_no].index.to_list():
            # getting p-values to determine significant level
            pval = s['summary'][reg_no].loc[var,'pval']
            param =  params_format.format(s['summary'][reg_no].loc[var,'mean'])
            # add significant level star mark to params
            if (pval <= 0.1) & (pval > 0.05):
                param = param + '*'
            elif (pval <= 0.05) & (pval > 0.01):
                param = param + '**'
            elif pval <= 0.01:
                param = param + '***'

            # filling param
            summary_table.loc[var, model_no] = param
            # filling t-statistics
            summary_table.loc['{}_t'.format(var), model_no] = tstat_format.format(s['summary'][reg_no].loc[var,'tstat'])

    # replacing `var_t` index with whitespace,
    summary_table.index = sum([[var, ''.format(var)] for var in full_xvars], [])

    # replace NaN with whitespace for readability
    return summary_table.fillna('')