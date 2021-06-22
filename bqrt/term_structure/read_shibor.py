#!/usr/bin/env python

import numpy as np
import pandas as pd
import glob
from sklearn.linear_model import LinearRegression


def read_shibor(path) -> pd.DataFrame:
    """Batch read Shibor xls files

    Batch read xls files downlaod from Shibor Data Services, with columns of following structure of "date, O/N, 1W, 2W, 1M, 3M, 6M, 9M, 1Y" . For example "myproject\\raw\\shibor"

    [Shibor Data Services](http://www.shibor.org/shibor/web/DataService.jsp) 

    Args:
        path (str): [file path containing Shibor xls files]
    """
    shibor_file_list = glob.glob(path+'\\*.xls')
    shibor_file_parts = [pd.read_excel(
        f, parse_dates=['日期']) for f in shibor_file_list]
    shibor_df = pd.concat(shibor_file_parts, ignore_index=True)

    return shibor_df


def rate_interp(date, days):
    """ term structure linear interpolation

    Return linear interpolation results of interest rate
    """
    X = np.array([[30, 90, 180, 270, 360]]).reshape(-1, 1)
    y = rate_df[rate_df['date'] == date][[
        '1m', '3m', '6m', '9m', '1y']].values.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    return reg.predict(np.array([[days]])).item()/100
