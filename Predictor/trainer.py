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
    return new_df_2


def main():
    
    path = "./train_data/GRU/output_24.xlsx"
    df = pd.read_excel(path)
    dataset = difference_df(df)
    X_series = dataset['X']
    Y_series = dataset['Y']
    Z_series = dataset['Z']

    model = auto_arima(X_series)
    model = model.fit()
    joblib.dump(model, './models/GRU/model_gru_X.pkl')

    model = auto_arima(Y_series)
    model = model.fit()
    joblib.dump(model, './models/GRU/model_gru_Y.pkl')

    model = auto_arima(Z_series)
    model = model.fit()
    joblib.dump(model, './models/GRU/model_gru_Z.pkl')

if __name__ == "__main__":
	main()
