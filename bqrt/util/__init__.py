#!/usr/bin/env python

import pandas as pd
import glob


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


def check_dates():
    print(3)


def check_entries():
    pass
