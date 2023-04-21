"""
Portfolio Optimization
--------------
"""

import numpy as np

def crra_ultility(r, gamma, bounds=None):

    if bounds:
        assert isinstance(bounds, tuple), 'bounds must be `(lower, upper)` tuple'

    if gamma ==1:
        utility = np.log(r)
    else:
        utility =  (r**(1-gamma))/ (1-gamma)

    if bounds:
        return np.clip([utility], bounds[0], bounds[1])[0]
    else:
        return utility


def mean_var_utility(r, sigma, gamma, bounds=(0,1)):
    assert isinstance(bounds,tuple), 'bounds must be `(lower, upper)` tuple'

    utility = r - 0.5 * gamma * (sigma**2)

    return np.clip([utility], bounds[0], bounds[1])[0]


def mean_var_utility(r, sigma, gamma, bounds=None):

    if bounds:
        assert isinstance(bounds, tuple), 'bounds must be `(lower, upper)` tuple'

    utility = r - 0.5 * gamma * (sigma**2)
    if bounds:
        return np.clip([utility], bounds[0], bounds[1])[0]
    else:
        return utility



def mean_var_opt_weight(r, sigma, gamma):

    return r / (gamma * sigma**2)




