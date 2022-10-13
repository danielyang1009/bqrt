#!/usr/bin/env python
from .date import *
from .stats import *

import pandas as pd
# import numpy as np


def set_option_display(rows:int=200, cols:int=50):
    """
    Set both maximum diplay rows and columns

    Parameters
    ----------
    rows : int, optional
        max rows, by default 200
    cols : int, optional
        max columns, by default 50
    """
    pd.set_option('display.max_rows', rows)
    pd.set_option('display.max_columns', cols)


def set_option_float(digits=4):
    """
    Set maximum number of digits to show

    Parameters
    ----------
    digits : int, optional
        number of digits to show, by default 4
    """

    float_format = '{:.'+str(digits)+'f}'
    pd.set_option('display.float_format', lambda x: float_format.format(x))


def flat_multi_idx(df:pd.DataFrame) -> list:
    """Flatten pandas multi-index after unstack
    `df.columns = flat_multi_idx(df)`

    Parameters
    ----------
    df : pd.DataFrame
        df which columns is multi-index

    Returns
    -------
    list
        flatten list of df columns
    """
    return ['_'.join(col).strip() for col in df.columns.values]


def hac_lag(df:pd.DataFrame):
    """Calculate three kind of estimate of HAC lag

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to calculate
    """
    print( 4*(df.shape[0]/100)**(2/9), 0.75*(df.shape[0])**(1/3)-1, (df.shape[0])**(1/4))


def slope_intercept(x1, y1, x2, y2):
    """ Testing liear inpterpolation

    Parameters
    ----------
    x1 : float
        x1
    y1 : float
        y1
    x2 : float
        x2
    y2 : float
        y2

    Returns
    -------
    float
        return linear interpolation slope and intercept
    """
    slope = (y2 - y1) / (x2 - x1)
    intercept = (x1 * y2 - x2 * y1) / (x1 - x2)
    return slope, intercept


def read_h5_folder(path) -> pd.DataFrame:
    """ Reading h5 file in a folder

    Reading all hdf file in 'path' (For example, 'raw/opt') and concatenate all file together to generate one single DataFrame. All file in the folder need to be in the same format(columns) to be concatenated.

    Parameters
    ----------
    path : string 
        file path, for exampe '/raw/50etf'

    Returns
    -------
    DataFrame
        return concatenated Dataframe
    """

    import glob

    h5_file_list = glob.glob(path + '/*.h5')
    h5_file_parts = [pd.read_hdf(f) for f in h5_file_list]
    h5_raw = pd.concat(h5_file_parts, ignore_index=True)
    return h5_raw


def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    """
    Plot time series

    Parameters
    ----------
    y : pd.Series
        time series to graph
    lags : [type], optional
        [description], by default None
    figsize : tuple, optional
        [description], by default (10, 8)
    style : str, optional
        [description], by default 'bmh'
    """

    import matplotlib.pyplot as plt
    import statsmodels.tsa.api as smt
    import statsmodels.api as sm
    import scipy.stats as scs
    
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 


def jupyter_memory():
    import sys

    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)