#!/usr/bin/env python
from .date import *
from .stats import *

import pandas as pd
# import numpy as np

def check_ptable(ptable, threshold=0.1, stars=False, columns=None):
    assert isinstance(ptable, pd.DataFrame), '应为DataFrame'
    # 如果未指定columns检查所有columns为float，指定则只检查指定列
    if columns == None:
        columns = ptable.columns
    assert all(ptable[columns].dtypes == float), '所有的列应为float'
    # 去除
    res = ptable[ptable <= threshold]
    # 精确两位
    res = res.round(2)
    # 打星
    if stars == True:
        res = res.applymap(lambda x: str(x)+''.join(['*' for t in [.01, .05, .1] if x<=t]))

    return res.astype(str).replace({'nan':''})


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


def jupyter_memory():
    import sys

    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)