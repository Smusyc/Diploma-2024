import random
from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd
import copy
import os
import re
import configparser
import sys
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from pmdarima.datasets import load_lynx
import joblib
import pickle

def difference_df(df):
    new_df = df.rolling(2).apply(lambda n: (np.array(n))[1] - (np.array(n))[0])
    new_df = new_df.fillna(0)
    indexes = new_df.index
    new_df.drop (index=indexes[0], axis= 0 , inplace= True )
    return new_df
    
def cascade_summ_df(new_df):
    new_df_2 = copy.deepcopy(new_df)
    new_df_index = new_df.index
    accum_row = new_df.loc[new_df_index[0]]
    for index, row in new_df.iterrows():
        accum_row += row
        new_df_2.loc[index] += accum_row
    
    indexes = new_df_2.index
    #new_df_2.drop(index=indexes[0], axis= 0 , inplace= True )
    return new_df_2


def main():

    #path = str(sys.argv[1])
    #path = "./train_data/ARIMA/dataset_for_ARIMA_trainer.xlsx"
    
    path = "./train_data/GRU/output_24.xlsx"
    '''
    if(path == ""):s
        path = "./train_data/ARIMA/dataset_for_ARIMA_trainer.xlsx"
    else:
        path = "./train_data/ARIMA/" + path
    '''
    df = pd.read_excel(path)
    
    #df = pd.read_excel("./train_data/dataset_for_ARIMA_trainer.xlsx")
    '''
    print(f'AUTOATIMA [dataset]: {sys.argv[1]}')
    

    print(f'AUTOATIMA [X_series]: ',df['X'][start:end].head())
    print(f'AUTOATIMA [Y_series]: ',df['Y'][start:end].head())
    print(f'AUTOATIMA [Z_series]: ',df['Z'][start:end].head())
    '''
    
    dataset = difference_df(df)
    
    X_series = dataset['X']
    
    Y_series = dataset['Y']
    
    Z_series = dataset['Z']

    #model = ARIMA(X_series, order=(2,1,0))
    model = auto_arima(X_series)
    model = model.fit()
    #save_model = model.save('./models/ARIMA/model_arima_X.pkl')
    joblib.dump(model, './models/GRU/model_gru_X.pkl')
    
    #model = ARIMA(Y_series, order=(2,1,0))
    model = auto_arima(Y_series)
    model = model.fit()
    #save_model = model.save('./models/ARIMA/model_arima_Y.pkl')
    joblib.dump(model, './models/GRU/model_gru_Y.pkl')

    
    #model = ARIMA(Z_series, order=(2,1,0))
    model = auto_arima(Z_series)
    model = model.fit()
    #save_model = model.save('./models/ARIMA/model_arima_Z.pkl')
    joblib.dump(model, './models/GRU/model_gru_Z.pkl')

if __name__ == "__main__":
	main()