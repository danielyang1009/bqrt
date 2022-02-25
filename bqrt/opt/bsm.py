#!/usr/bin/env python

from numpy import exp, log, sqrt, nan
from scipy.stats import norm
from scipy import optimize


def bsm_d1(S,K,r,tau,sigma):
    return (log(S/K) + (r+0.5*sigma**2)*tau) / (sigma*sqrt(tau))


def bsm_d2(S,K,r,tau,sigma):
    return (log(S/K) + (r-0.5*sigma**2)*tau) / (sigma*sqrt(tau))


def bsm_call_delta(S,K,r,tau,sigma):
    d1 = (log(S/K) + (r+0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    return norm.cdf(d1)


def bsm_put_delta(S,K,r,tau,sigma):
    d1 = (log(S/K) + (r+0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    return norm.cdf(d1) - 1


def bsm_gamma(S,K,r,tau,sigma) -> float:
    d1 = (log(S/K) + (r+0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    return norm.pdf(d1) / (S*sigma*sqrt(tau))


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
    return optimize.newton(lambda x: bsm_price(S,K,r,tau,x,cp_flag) - cp_mkt_value, 0.5, maxiter=100, disp=True)


def bsm_vega(S,K,r,tau,sigma):
    d1 = (log(S/K) + (r+0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    return S*norm.pdf(d1)*sqrt(tau)


def bsm_theta(S,K,r,tau,sigma,cp_flag) -> float:
    d1 = (log(S/K) + (r+0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    d2 = d1 - sigma*sqrt(tau)
    if cp_flag == 'C':
        return -(S*norm.pdf(d1)*sigma)/(2*sqrt(tau)) - r*K*exp(-r*tau)*norm.cdf(d2)
    elif cp_flag == 'P':
        return -(S*norm.pdf(d1)*sigma)/(2*sqrt(tau)) + r*K*exp(-r*tau)*norm.cdf(-d2)


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