#!/usr/bin/env python
from .date import *
from .stats import *

import pandas as pd
# import numpy as np


def set_notebook(max_rows:int=200, max_columns:int=50, float_digi=4):
    """
    1. Set both maximum diplay rows and columns
    2. Set maximum number of digits to show

    Parameters
    ----------
    max_rows : int, optional
        max rows, by default 200
    max_columns : int, optional
        max columns, by default 50
    float_digi : int, optional
        number of digits to show, by default 4
    """
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_columns)

    float_format = '{{:.{}f}}'.format(float_digi)
    pd.set_option('display.float_format', lambda x: float_format.format(x))


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