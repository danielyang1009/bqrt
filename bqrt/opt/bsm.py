#!/usr/bin/env python

from numpy import exp, log, sqrt, nan
from scipy.stats import norm
from scipy import optimize

def black_d1(F,K,tau,sigma):
    return (log(F/K)+(0.5*sigma**2)*tau) / (sigma*sqrt(tau))


def black_d2(F,K,tau,sigma):
    return (log(F/K)-(0.5*sigma**2)*tau) / (sigma*sqrt(tau))


def black_call_delta(F,K,r,tau,sigma) -> float:
    return exp(-r*tau) * norm.cdf(black_d1(F,K,tau,sigma))


def black_put_delta(F,K,r,tau,sigma) -> float:
    return -exp(-r*tau) * norm.cdf(-black_d1(F,K,tau,sigma))


def black_call_bound(F,K,r,tau,call_price) -> bool:
    lower = max((F-K)*exp(-r*tau),0)
    upper = F * exp(-r*tau)
    return (call_price >= lower)&(call_price <= upper)


def black_put_bound(F,K,r,tau,put_price) -> bool:
    lower = max((K-F)*exp(-r*tau),0)
    upper = K * exp(-r*tau)
    return (put_price >= lower)&(put_price <= upper)


def black_price(F,K,r,tau,sigma,cp_flag):
    d1 = (log(F/K)+(0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    d2 = (log(F/K)-(0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    if cp_flag == 'C':
        return exp(-r*tau)*(F*norm.cdf(d1) - K*norm.cdf(d2))
    elif cp_flag == 'P':
        return exp(-r*tau)*(K*norm.cdf(-d2) - F*norm.cdf(-d1))


def black_impl_vol(F,K,r,tau,cp_mkt_value,cp_flag):
    try:
        return optimize.newton(lambda x: black_price(F,K,r,tau,x,cp_flag) - cp_mkt_value, 0.5)
    except:
        return nan


def black_vega(F,K,T,sigma):
    d1 = (log(F/K)+(0.5*sigma**2)*T) / (sigma*sqrt(T))
    vega = F*norm.pdf(d1)*sqrt(T)
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


def bsm_d1(S,K,r,tau,sigma):
    return (log(S/K) + (r+0.5*sigma**2)*tau) / (sigma*sqrt(tau))


def bsm_d1(S,K,r,tau,sigma):
    return (log(S/K) + (r-0.5*sigma**2)*tau) / (sigma*sqrt(tau))


def bsm_call_delta(S,K,r,tau,sigma):
    return norm.cdf(bsm_d1(S,K,r,tau,sigma))


def bsm_put_delta(S,K,r,tau,sigma):
    return -norm.cdf(-bsm_d1(S,K,r,tau,sigma))


def bsm_call_bound(S,K,r,tau,call_price) -> bool:
    lower = max(S - K*exp(-r*tau),0)
    upper = S
    return (call_price >= lower)&(call_price <= upper)


def bsm_put_bound(S,K,r,tau,put_price) -> bool:
    lower = max(K*exp(-r*tau) - S,0)
    upper = K * exp(-r*tau)
    return (put_price >= lower)&(put_price <= upper)


def bsm_price(S,K,r,tau,sigma,cp_flag):
    d1 = (log(S/K) + (r+0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    d2 = d1 - sigma*sqrt(tau)
    if cp_flag == 'C':
        return S*norm.cdf(d1) - K*exp(-r*tau)*norm.cdf(d2)
    elif cp_flag == 'P':
        return K*exp(-r*tau)*norm.cdf(-d2) - S*norm.cdf(-d1)


def bsm_impl_vol(S,K,r,tau,cp_mkt_value,cp_flag):
    try:
        return optimize.newton(lambda x: bsm_price(S,K,r,tau,x,cp_flag) - cp_mkt_value, 0.5)
    except:
        return nan


def bsm_vega(S,K,r,T,sigma):
    d1 = (log(S/K)+(r+0.5*sigma**2)*T) / (sigma*sqrt(T))
    vega = S*norm.pdf(d1, 0.0, 1.0)*sqrt(T)
    return vega


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