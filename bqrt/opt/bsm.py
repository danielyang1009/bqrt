#!/usr/bin/env python

from numpy import exp, log, sqrt, nan
from scipy.stats import norm
from scipy import optimize


def black_price(F,K,r,T,sigma,cp_flag):
    d1 = (log(F/K)+(0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = (log(F/K)-(0.5*sigma**2)*T) / (sigma*sqrt(T))
    if cp_flag == 'C':
        return exp(-r*T)*(F*norm.cdf(d1) - K*norm.cdf(d2))
    elif cp_flag == 'P':
        return exp(-r*T)*(K*norm.cdf(-d2) - F*norm.cdf(-d1))


def bsm_price(S,K,r,T,sigma,cp_flag):
    d1 = (log(S/K) + (r+0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if cp_flag == 'C':
        return S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
    elif cp_flag == 'P':
        return K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def black_vega(F,K,r,T,sigma):
    d1 = (log(F/K)+(0.5*sigma**2)*T) / (sigma*sqrt(T))
    vega = F*norm.pdf(d1)*sqrt(T)
    return vega


def bsm_vega(S,K,r,T,sigma):
    d1 = (log(S/K)+(r+0.5*sigma**2)*T) / (sigma*sqrt(T))
    vega = S*norm.pdf(d1, 0.0, 1.0)*sqrt(T)
    return vega

# old version 
def black_impl_vol0(F,K,r,T,cp_mkt_value,cp_flag):
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5
    sigma = 0.5
    for _ in range(MAX_ITERATIONS):
        # every iter calculate a new pair of price and vega
        price = black_price(F,K,r,T,sigma,cp_flag)
        vega = black_vega(F,K,r,T,sigma)
        diff = price - cp_mkt_value
        if (abs(diff) < PRECISION):
            return sigma
        # divide by zero error
        elif vega != 0:
            sigma = sigma - diff/vega
        elif vega == 0:
            return -1


def black_impl_vol(F,K,r,T,cp_mkt_value,cp_flag):
    try:
        return optimize.newton(lambda x: black_price(F,K,r,T,x,cp_flag) - cp_mkt_value, 0.5)
    except:
        return nan


# old version
def bsm_impl_vol0(S,K,r,T,cp_mkt_value,cp_flag):
    MAX_ITERATIONS = 100
    PRECISION = 1.0e-5
    sigma = 0.5
    for _ in range(MAX_ITERATIONS):
        # every iter calculate a new pair of price and vega
        price = bsm_price(S,K,r,T,sigma,cp_flag)
        vega = bsm_vega(S,K,r,T,sigma)
        diff = price - cp_mkt_value
        if (abs(diff) < PRECISION):
            return sigma
        # divide by zero error
        elif vega != 0:
            sigma = sigma - diff/vega
        elif vega == 0:
            return -1


def bsm_impl_vol(S,K,r,T,cp_mkt_value,cp_flag):
    return optimize.newton(lambda x: bsm_price(S,K,r,T,x,cp_flag) - cp_mkt_value, 0.5)


def test1():
    print(3)