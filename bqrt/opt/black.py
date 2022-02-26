#!/usr/bin/env python

from numpy import exp, log, sqrt
from scipy.stats import norm
from scipy import optimize


def black_d1(F,K,tau,sigma):
    return (log(F/K) + (0.5*sigma**2)*tau) / (sigma*sqrt(tau))


def black_d2(F,K,tau,sigma):
    return (log(F/K) - (0.5*sigma**2)*tau) / (sigma*sqrt(tau))


def black_delta(F,K,r,tau,sigma,cp_flag) -> float:
    d1 = (log(F/K) + (0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    if cp_flag == 'C':
        return exp(-r*tau) * norm.cdf(d1)
    elif cp_flag == 'P':
        return -exp(-r*tau) * norm.cdf(-d1)


def black_call_bound(F,K,r,tau,call_price) -> bool:
    lower = max((F-K)*exp(-r*tau),0)
    upper = F * exp(-r*tau)
    return (call_price >= lower)&(call_price <= upper)


def black_put_bound(F,K,r,tau,put_price) -> bool:
    lower = max((K-F)*exp(-r*tau),0)
    upper = K * exp(-r*tau)
    return (put_price >= lower)&(put_price <= upper)


def black_price(F,K,r,tau,sigma,cp_flag):
    d1 = (log(F/K) + (0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    d2 = (log(F/K) - (0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    if cp_flag == 'C':
        return exp(-r*tau)*(F*norm.cdf(d1) - K*norm.cdf(d2))
    elif cp_flag == 'P':
        return exp(-r*tau)*(K*norm.cdf(-d2) - F*norm.cdf(-d1))


def black_impl_vol(F,K,r,tau,cp_mkt_value,cp_flag):
    return optimize.newton(lambda x: black_price(F,K,r,tau,x,cp_flag) - cp_mkt_value, 0.5, maxiter=100, disp=True)


def black_vega(F,K,tau,sigma):
    d1 = (log(F/K)+(0.5*sigma**2)*T) / (sigma*sqrt(tau))
    return F*norm.pdf(d1)*sqrt(tau)


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