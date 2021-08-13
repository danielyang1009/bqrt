#!/usr/bin/env python

import numpy as np
from scipy.stats import norm


def black_price(F,K,r,T,sigma,cp_flag):
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(F / K) - (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if cp_flag == 'C':
        return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    elif cp_flag == 'P':
        return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def black_vega(F,K,r,T,sigma):
    d1 = (np.log(F / K) + (0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = F * norm.pdf(d1) * np.sqrt(T)
    return vega


# old version, will encounter devide by zero error
def black_impv_0(F,K,r,T,value,cp_flag):
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(MAX_ITERATIONS):
        # every iter calculate a new pair of price and vega
        price = black_price(F,K,r,T,sigma,cp_flag)
        vega = black_vega(F,K,r,T,sigma)
        diff = price - value
        if (abs(diff) < PRECISION):
            return sigma
        sigma = sigma - diff/vega
        print(sigma)
    return sigma


def black_impl_vol(F,K,r,T,value,cp_flag):
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(MAX_ITERATIONS):
        # every iter calculate a new pair of price and vega
        price = black_price(F,K,r,T,sigma,cp_flag)
        vega = black_vega(F,K,r,T,sigma)
        diff = price - value
        if (abs(diff) < PRECISION):
            return sigma
        # divide by zero error
        elif vega != 0:
            sigma = sigma - diff/vega
        elif vega == 0:
            return -1