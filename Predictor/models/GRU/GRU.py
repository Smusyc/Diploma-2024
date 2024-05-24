import sys
import pandas as pd
import numpy as np
import os
import copy
import math
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

def difference_df(df):
    new_df = pd.DataFrame()

    new_df['X'] = (df['X'].rolling(2)).apply(lambda n: (np.array(n))[1] - (np.array(n))[0])
    new_df['Y'] = (df['Y'].rolling(2)).apply(lambda n: (np.array(n))[1] - (np.array(n))[0])
    new_df['Z'] = (df['Z'].rolling(2)).apply(lambda n: (np.array(n))[1] - (np.array(n))[0])
    new_df['time'] = (df['time'].rolling(2)).apply(lambda n: (np.array(n))[1] - (np.array(n))[0])
    new_df['trace'] = (df['trace'].rolling(2)).apply(lambda n: (np.array(n))[1])
    new_df = new_df.fillna(0)
    print(f"indexes = {(new_df.index)[-1]}")
    for index in range(1, (new_df.index)[-1]):
        if (new_df['trace'].iloc[index] != new_df['trace'].iloc[index+1]):
            new_df['X'].iloc[index+1] = new_df['X'].iloc[index]
            new_df['Y'].iloc[index+1] = new_df['Y'].iloc[index]
            new_df['Z'].iloc[index+1] = new_df['Z'].iloc[index]
            new_df['time'].iloc[index+1] = new_df['time'].iloc[index]
    
def cascade_summ_df(new_df):
    new_df_2 = copy.deepcopy(new_df)
    new_df_index = new_df.index
    accum_row = new_df.loc[new_df_index[0]]
    for index, row in new_df.iterrows():
        accum_row += row
        new_df_2.loc[index] += accum_row
    
    indexes = new_df_2.index
    return new_df_2


def main():
    print('=======================================================')
    start = int(sys.argv[2])
    print(f'GRU [start]: {sys.argv[2]}')
    
    end = int(sys.argv[3])
    print(f'GRU [end]: {sys.argv[3]}')
    
    df = pd.read_excel(sys.argv[1])
    df_for_training = pd.read_excel(sys.argv[1])
    print(f'GRU [dataset]: {sys.argv[1]}')
    

    print(f'GRU [X_series]: ',df['X'][start:end].head())
    print(f'GRU [Y_series]: ',df['Y'][start:end].head())
    print(f'GRU [Z_series]: ',df['Z'][start:end].head())
    
    diff_dataset_from_df = difference_df(df)
    diff_dataset_for_training = difference_df(df_for_training)
    
    X_series = diff_dataset_for_training['X']
    print(f'GRU [X_series_difference]: {X_series.head()}; Index = {(X_series.index)[0]}')
    
    Y_series = diff_dataset_for_training['Y']
    print(f'GRU [Y_series]: {Y_series.head()}; Index = {(Y_series.index)[0]}')
    
    Z_series = diff_dataset_for_training['Z']
    print(f'GRU [Z_series]: {Z_series.head()}; Index = {(Z_series.index)[0]}')
    
    model = keras.models.load_model('./models/GRU/GRU_L2_U50_E5_v1_X')
    X_series_pred = model.predict(start=start, end=end-1)
    print(f'GRU [X_series_pred]: {X_series_pred.size}')

    model = keras.models.load_model('./models/GRU/GRU_L2_U50_E5_v1_Y')
    Y_series_pred = model.predict(start=start, end=end-1)
    print(f'GRU [Y_series_pred]: {Y_series_pred.size}')
    
    model = keras.models.load_model('./models/GRU/GRU_L2_U50_E5_v1_Z')
    Z_series_pred = model.predict(start=start, end=end-1)
    print(f'GRU [Z_series_pred]: {Z_series_pred.size}')
    
    
    time_period = diff_dataset_from_df['time'].mode()[0]
    
    data = {'X': X_series_pred,
            'Y': Y_series_pred,
            'Z': Z_series_pred
            }

    result_df = pd.DataFrame(data)
    result_df = result_df.fillna(0)
    result_df['time'] = time_period
    result_df['trace'] = df['trace'].mode()[0]
    
    result_df_exel_2 = cascade_summ_df(result_df)
    print(f'====GRU cascade result_df_exel_2: ',result_df_exel_2.head())
    
    result_df_exel_2['X'] = result_df_exel_2['X'].apply(lambda n: n + df['X'].iloc[start])
    result_df_exel_2['Y'] = result_df_exel_2['Y'].apply(lambda n: n + df['Y'].iloc[start])
    result_df_exel_2['Z'] = result_df_exel_2['Z'].apply(lambda n: n + df['Z'].iloc[start])
    result_df_exel_2['time'] = result_df_exel_2['time'].apply(lambda n: n + df['time'].iloc[start])
    result_df_exel_2['trace'] = df['trace'].iloc[start]
    result_df_2=result_df_exel_2
    
    if(os.path.exists("./models/temp_dataset.xlsx")):
        os.remove("./models/temp_dataset.xlsx")
    result_df_2.to_excel("./models/temp_dataset.xlsx")
    
    return
    
    
if __name__ == '__main__':
    print(__name__)
    main()
