#!/usr/bin/env python

import numpy as np
import pandas as pd
import glob
from scipy.interpolate import interp1d

# from sklearn.linear_model import LinearRegression


def read_shibor(path) -> pd.DataFrame:
    """Batch read Shibor xls files

    Batch read xls files downlaod from Shibor Data Services, with columns of following structure of "date, O/N, 1W, 2W, 1M, 3M, 6M, 9M, 1Y" . For example "myproject\\raw\\shibor"

    [Shibor Data Services](http://www.shibor.org/shibor/web/DataService.jsp) 

    Args:
        path (str): [file path containing Shibor xls files]
    """

    shibor_file_list = glob.glob(path + "\\*.xls")
    shibor_file_parts = [
        pd.read_excel(f, parse_dates=[0]) for f in shibor_file_list
    ]
    shibor_df = pd.concat(shibor_file_parts, ignore_index=True)
    shibor_df["date"] = pd.to_datetime(
        shibor_df["date"].dt.date)  # keep only date
    return shibor_df


def shibor_linear_interp(shibor_df, date, maturity) -> float:
    """Linear interpolation of Shibor term structure

    Linear interpolation of Shibor Data Services xls files

    Args:
        shibor_df ([pd.DataFrame]): [Datafram from read_shibor]
        date ([str]): [Date in string]
        maturity ([type]): [maturity in year]

    Returns:
        [float]: [linear interpolation result from input date and maturity]
    """
    maturity_list = [0, 7.0 / 365, 14.0 / 365, 1.0 / 12, 0.25, 0.5, 0.75, 1]
    term_stucture = shibor_df[shibor_df['date'] ==
                              date].values.flatten().tolist()[1:]
    return np.interp(maturity, maturity_list, term_stucture)