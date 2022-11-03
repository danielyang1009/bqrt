"""
Portfolio Sort
--------------
"""
import pandas as pd
import numpy as np
from tqdm.notebook import trange, tqdm


def port_sort(data, time, entity, ret, criteria, num=10, weight=None, min_entities=400):
    
    # will alter data, set copy
    df = data.copy()
    
#     forward criteria
    df[criteria] = df.groupby(entity)[criteria].shift(1)
    # lag return
    # df[ret] = df.groupby(entity)[ret].shift(-1)
    df = df.dropna().reset_index(drop=True)
    
    # checking min entities
    size_list = df.groupby(time).size().to_frame().reset_index()
    size_list.columns = [time,'size']
    print('Skip days with less {} entities'.format(min_entities))
    print(size_list[size_list['size'] < min_entities][time])
    time_filter = size_list[size_list['size'] > min_entities][time].sort_values()
    
    # setup dataframe, date as index
    result = pd.DataFrame(np.zeros((len(time_filter),num)))
    result.index = time_filter
    result.columns = np.arange(1,num+1)
    
    for date in tqdm(result.index):
        ref = df[df[time] == date].copy()

        # accending 10 -> biggest criteria group
        ref.loc[:,'rank'] = pd.qcut(ref[criteria], num, labels=np.arange(1,num+1))
        for rank in np.arange(1,num+1):
            if weight == None:
                result.loc[result.index == date, rank] = ref[ref['rank'] == rank][ret].mean()
            else:
                result.loc[result.index == date, rank] = ref[ref['rank'] == rank][ret] * ref[ref['rank'] == rank][criteria] / ref[ref['rank'] == rank][criteria].sum()
    
    result.columns = [str(i) for i in np.arange(1,num+1)]
    result['{}-1'.format(num)] = result[str(num)] - result['1']
    return result

    
def port_plot(pa_result, skip=True):
    if skip:
        (pa_result.drop(columns=['10-1'])+1).cumprod().plot(figsize=(12,6),grid=True)
    else:
        (pa_result+1).cumprod().plot(figsize=(12,6),grid=True)