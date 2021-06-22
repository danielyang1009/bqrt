#!/usr/bin/env python

import numpy as np 
from sklearn.linear_model import LinearRegression


def rate_interp(date,days):
    """
    Return linear interpolation results of interest rate
    """
    X = np.array([[30,90,180,270,360]]).reshape(-1,1)
    y = rate_df[rate_df['date'] == date][['1m','3m','6m','9m','1y']].values.reshape(-1,1)
    reg = LinearRegression().fit(X,y)
    return reg.predict(np.array([[days]])).item()/100