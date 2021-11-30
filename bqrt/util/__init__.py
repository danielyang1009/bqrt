#!/usr/bin/env python

import pandas as pd
import glob
from .dist import *
from .date import *


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
    h5_file_list = glob.glob(path + '/*.h5')
    h5_file_parts = [pd.read_hdf(f) for f in h5_file_list]
    h5_raw = pd.concat(h5_file_parts, ignore_index=True)
    return h5_raw


import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

def tsplot(y, lags=None, figsize=(10, 8), style='bmh'):
    """ Plot time series

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


def check_dates():
    print(3)


def check_entries():
    pass
