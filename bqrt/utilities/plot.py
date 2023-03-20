"""
Plot
----
"""

import pandas as pd

def plot_cumprod(df,figsize=(8,4)):

    (df+1).cumprod().plot(figsize=figsize)