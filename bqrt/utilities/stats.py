from numpy import exp,log, sqrt, pi
import pandas as pd
# from scipy import stats
from scipy.special import erf
# from scipy.stats import norm
# from scipy.stats import lognorm

# Reference
# https://en.wikipedia.org/wiki/Normal_distribution
# https://en.wikipedia.org/wiki/Log-normal_distribution
# https://en.wikipedia.org/wiki/Error_function
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
# https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.stats.lognorm.html


def describe(df:pd.DataFrame, params_digi:int=4, tstat_digi:int=2):
    """
    Create pandas describe like dataframe, contains skewness and kurtosis, t-statistics and p-value

    Need to have `dropna` first.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame object to be described, exclude columns not to be described
    digits : int
        number of digits to show

    Returns
    -------
    result : pd.DataFrame
        DataFrame stats
    """
    import scipy.stats as stats

    result_list = []
    params_format = '{{:.{}f}}'.format(params_digi)
    tstat_format = '{{:.{}f}}'.format(tstat_digi)

    # only include `float` dtype
    df = df.select_dtypes(include='float')

    for col in df.columns.to_list():
        des = stats.describe(df[col], nan_policy='omit')
        ttest = stats.ttest_1samp(df[col],0,nan_policy='omit')
        parts = {
            'Count': '{}'.format(int(des.nobs)),
            'Mean': params_format.format(des.mean),
            'SD': params_format.format(sqrt(des.variance)),
            # 'var': params_format.format(des.variance),
            'Min': params_format.format(des.minmax[0]),
            '5%': params_format.format(df[col].quantile(0.05)),
            '25%': params_format.format(df[col].quantile(0.25)),
            '50%': params_format.format(df[col].quantile(0.50)),
            '75%': params_format.format(df[col].quantile(0.75)),
            '95%': params_format.format(df[col].quantile(0.95)),
            'Max': params_format.format(des.minmax[1]),
            'Skew': params_format.format(des.skewness),
            'Kurt': params_format.format(des.kurtosis),
            'T-stat': tstat_format.format(ttest[0]),
            'P-val': tstat_format.format(ttest[1])
        }
        result_list.append(parts)
        result = pd.DataFrame(result_list).T
    result.columns = df.columns
    return result.T


def norm_pdf(x,mu,sigma):
    """Normal distribution PDF

    Parameters
    ----------
    x : float
    mu : float
    sigma : float

    Returns
    -------
    float
        Probability from normal distribution evalutad at x
    """
    # calcuated from scipy package
    # norm.pdf(x,mu,sigma)
    return 1/(sigma*sqrt(2*pi)) * exp(-(x-mu)**2/(2*sigma**2))


def norm_cdf(x,mu,sigma):
    """Normal distribution CDF

    Parameters
    ----------
    x : float
    mu : float
    sigma : float

    Returns
    -------
    float
        Culmulated probability from normal distribution evalutad from -inf to x
    """
    # calcuated from scipy package
    # norm.cdf(x,mu,sigma)
    return 0.5*(1 + erf((x-mu)/(sigma*sqrt(2))))


def logn_pdf(x,mu,sigma):
    """Log-normal distribution PDF

    Parameters
    ----------
    x : float
    mu : float
    sigma : float

    Returns
    -------
    float
        Probability from normal distribution evalutad at x
    """
    # calculated from scipy package, s=sigma, loc=0, scale=exp(mu)
    # lognorm.pdf(x,mu,sigma,0,np.exp(mu))
    return 1/(x*sigma*sqrt(2*pi)) * exp(-(log(x)-mu)**2/(2*sigma**2))


def logn_cdf(x,mu,sigma):
    """Log-normal distribution CDF

    Parameters
    ----------
    x : float
    mu : float
    sigma : float

    Returns
    -------
    float
        Culmulated probability from log-normal distribution evalutad from -inf to x
    """
    # calculated from scipy package, s=sigma, loc=0, scale=exp(mu)
    # lognorm.cdf(x,sigma,0,np.exp(mu))
    return 0.5*(1 + erf((log(x)-mu)/(sigma*sqrt(2))))


def norm_mean(logn_mean,logn_var):
    """Normal distribution mean or mu, calculated from log-normal mean and variance

    Parameters
    ----------
    logn_mean : float
        log-normal mean
    logn_var : float
        log-normal variance

    Returns
    -------
    float
        Normal distribution mean or mu
    """
    return log(logn_mean) - 0.5*log(1+logn_mean/logn_var**2)


def norm_var(logn_mean, logn_var):
    """Normal distribution variance or sigma^2, calculated from log-normal mean and variance

    Parameters
    ----------
    logn_mean : float
        log-normal mean
    logn_var : float
        log-normal variance

    Returns
    -------
    float
        Normal distribution variance or sigma^2
    """
    return log(1+logn_mean/logn_var**2)


def logn_mean(norm_mean,norm_var):
    """Log-normal distribution mean or mu, calculated from normal mean and variance

    Parameters
    ----------
    norm_mean : float
        normal mean
    norm_var : float
        normal variance

    Returns
    -------
    float
        Log-normal distribution mean or mu
    """
    return exp(norm_mean + norm_var/2)


def logn_var(norm_mean, norm_var):
    """Log-normal distribution variance or sigma^2, calculated from normal mean and variance

    Parameters
    ----------
    norm_mean : float
        normal mean
    norm_var : float
        normal variance

    Returns
    -------
    float
        Log-normal distribution variance or sigma^2
    """
    return exp(2*norm_mean + norm_var) * (exp(norm_var)-1)