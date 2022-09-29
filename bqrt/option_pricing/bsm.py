#!/usr/bin/env python

from numpy import exp, log, sqrt, nan
from scipy.stats import norm


def bsm_d1(S,K,r,tau,sigma):
    return (log(S/K) + (r+0.5*sigma**2)*tau) / (sigma*sqrt(tau))


def bsm_d2(S,K,r,tau,sigma):
    return (log(S/K) + (r-0.5*sigma**2)*tau) / (sigma*sqrt(tau))


def bsm_delta(S,K,r,tau,sigma,cp_flag) -> float:
    d1 = (log(S/K) + (r+0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    if cp_flag == 'C':
        return norm.cdf(d1)
    elif cp_flag == 'P':
        return norm.cdf(-d1)


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


def bsm_price(S,K,r,tau,sigma,cp_flag:str) -> float:
    """
    Black–Scholes–Merton model for european option pricing

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
        bsm model european option price

    Examples
    --------
    >>> bsm_price(52,50,0.05,0.5,0.12,'C')
    3.7881
    >>> bsm_price(52,50,0.05,0.5,0.12,'P')
    0.5536

    Reference
    ---------
    [1] Black, Fischer, and Myron Scholes, 1973, The Pricing of Options and Corporate Liabilities, Journal of Political Economy 81, 637–654.
    [2] Merton, Robert C., 1973, Theory of Rational Option Pricing, The Bell Journal of Economics and Management Science 4, 141–183.
    [3] Hull, John, 2017, Options, Futures, and Other Derivatives. 10th edition. (Pearson, New York, NY).

    """
    d1 = (log(S/K) + (r+0.5*sigma**2)*tau) / (sigma*sqrt(tau))
    d2 = d1 - sigma*sqrt(tau)
    if cp_flag == 'C':
        return S*norm.cdf(d1) - K*exp(-r*tau)*norm.cdf(d2)
    elif cp_flag == 'P':
        return K*exp(-r*tau)*norm.cdf(-d2) - S*norm.cdf(-d1)


def bsm_impl_vol(S,K,r,tau,cp_mkt_value,cp_flag):
    from scipy import optimize

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


# old version newton's method root-finding
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