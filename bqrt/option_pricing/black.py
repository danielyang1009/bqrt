#!/usr/bin/env python

from numpy import exp, log, sqrt
from scipy.stats import norm


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
    """
    Black model or Black-76 model for european option pricing

    Parameters
    ----------
    S : float
        spot price
    K : float
        strike price
    r : float
        risk-free rakte
    tau : _type_
        time to maturity (in year)
    sigma : float
        volatility
    cp_flag : string
        'C' for european call, 'P' for european put

    Returns
    -------
    float
        black model european option price

    Reference
    ---------
    [1] Black, Fischer, 1976, The Pricing of Commodity Contracts, Journal of Financial Economics 3, 167–179.
    [2] Hull, John, 2017, Options, Futures, and Other Derivatives. 10th edition. (Pearson, New York, NY).
    """

    d1 = (log(F/K) + (0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    d2 = (log(F/K) - (0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    if cp_flag == 'C':
        return exp(-r*tau)*(F*norm.cdf(d1) - K*norm.cdf(d2))
    elif cp_flag == 'P':
        return exp(-r*tau)*(K*norm.cdf(-d2) - F*norm.cdf(-d1))


def black_impl_vol(F,K,r,tau,cp_mkt_value,cp_flag):
    from scipy import optimize

    return optimize.newton(lambda x: black_price(F,K,r,tau,x,cp_flag) - cp_mkt_value, 0.5, maxiter=100, disp=True)


def black_vega(F,K,tau,sigma):
    d1 = (log(F/K)+(0.5*sigma**2)*T) / (sigma*sqrt(tau))
    return F*norm.pdf(d1)*sqrt(tau)


# old version newton's method root-finding
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