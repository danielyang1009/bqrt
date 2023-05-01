from .commom import *
# from .date import *
# from .stats import *
# from .testing import *

import pandas as pd
# import numpy as np


def popview(df):

    from IPython.display import HTML

    css = """<style>
    table { border-collapse: collapse; border: 3px solid #eee; }
    table tr th:first-child { background-color: #eeeeee; color: #333; font-weight: bold }
    table thead th { background-color: #eee; color: #000; }
    tr, th, td { border: 1px solid #ccc; border-width: 1px 0 0 1px; border-collapse: collapse;
    padding: 3px; font-family: monospace; font-size: 10px }</style>
    """
    s  = '<script type="text/Javascript">'
    s += 'var win = window.open("", "Title", "toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=780, height=200, top="+(screen.height-400)+", left="+(screen.width-840));'
    s += 'win.document.body.innerHTML = \'' + (df.to_html() + css).replace("\n",'\\') + '\';'
    s += '</script>'
    return(HTML(s+css))


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


def jupyter_memory():
    import sys

    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

    # Get a sorted list of the objects and their sizes
    sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)