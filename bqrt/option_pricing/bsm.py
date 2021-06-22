#!/usr/bin/env python

import numpy as np
from scipy.stats import norm


def bsm_price(S,K,r,T,sigma,cp_flag):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if cp_flag == 'C':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif cp_flag == 'P':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bsm_vega(S,K,r,T,sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
    return vega


def bsm_impl_vol(S,K,r,T,value,cp_flag):
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(MAX_ITERATIONS):
        # every iter calculate a new pair of price and vega
        price = bls_price(S,K,r,T,sigma,cp_flag)
        vega = bls_vega(S,K,r,T,sigma)
        diff = price - value
        if (abs(diff) < PRECISION):
            return sigma
        # divide by zero error
        elif vega != 0:
            sigma = sigma - diff/vega
        elif vega == 0:
            return -1