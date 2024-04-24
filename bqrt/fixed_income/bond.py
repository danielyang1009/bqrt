import numpy as np
import pandas as pd
import re
from scipy.optimize import minimize
from tqdm.notebook import tqdm

# 将中问1个月等，变为1m
def bond_col_cleaning(col_list, prefix=''):
    result = []
    for text in col_list:
        text = text.replace('年','y').replace('个月','m').replace('月','m').replace('天','d')
        text = re.findall(r'(\d+d|\d+m|\d+y)', text)
        if text: # 非空
            result.append(prefix + text[0])
    return result

def str_to_years(term):
    if term == '0y':
        result = 1/365
    else:
        number, unit = int(term[:-1]), term[-1]
        if unit == 'd':
            result = number / 365
        if unit == 'm':
            result = number / 12
        if unit == 'y':
            result = number
    return result


# 输入可以是DataFrame也可以是column
def col_to_years(col_list):
    return [str_to_years(term) for term in col_list]


# in years, including endpoints
def select_term(col_list, min_term, max_term):
    result = []
    for term in col_list:
        if (str_to_years(term) >= min_term) & (str_to_years(term) <= max_term):
            result.append(term)
    return result


# Diebold-Li 2006 Model
def b2_coeff(tau, lambd):
    return (1 - np.exp(-lambd*tau)) / (lambd*tau)


def b3_coeff(tau, lambd):
    return b2_coeff(tau, lambd) - np.exp(-lambd*tau)

def dl_model(tau, lambd, b1, b2, b3):
    y = b1 + b2*b2_coeff(tau, lambd) + b3*b3_coeff(tau, lambd)
    return y

def dl_objfunc(params, tau, y_obse):
    b1, b2, b3, lambd = params
    y_pred = dl_model(tau, lambd, b1, b2, b3)
    return np.sum((y_obse - y_pred)**2)


# with datetime index, wind-like col '0y, 1m, 2m, ...'
def dl_optimization(df:pd.DataFrame, init_params=[]):

    result = []
    terms = np.array(col_to_years(df.columns))
    for date in tqdm(df.index):
        # 取利率期限结构
        y_obse = df.loc[date, :].values

        opt_result = minimize(dl_objfunc, init_params, args=(terms, y_obse), method='SLSQP', options={'maxiter':100})

        b1, b2, b3, lambd = opt_result.x
        sucess = opt_result.success

        result.append([date,b1,b2,b3,lambd,sucess])

    return result