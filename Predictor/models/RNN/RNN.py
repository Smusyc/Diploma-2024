import tensorflow as tf
import sys
import pandas as pd
import numpy as np
import os
import copy
import math
import keras

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import GRU, Bidirectional
from keras.optimizers import SGD
from sklearn import metrics
from sklearn.metrics import mean_squared_error


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
    print('=======================================================')
    start = int(sys.argv[2])
    end = int(sys.argv[3])
    
    df = pd.read_excel(sys.argv[1])
    df_for_training = df
    
    diff_dataset_from_df = difference_df(df)
    diff_dataset_for_training = difference_df(df_for_training)
    
    X_series = diff_dataset_for_training['X']
    Y_series = diff_dataset_for_training['Y']
    Z_series = diff_dataset_for_training['Z']
    
    scaler = MinMaxScaler(feature_range=(0,1))
    dataset_test = X_series[start:end]
    dataset_test = np.reshape(dataset_test, (-1,1))
    scaled_test = scaler.fit_transform(dataset_test)
    
    X_test = []
    y_test = []
    for i in range(50, len(scaled_test)):
        X_test.append(scaled_test[i-50:i, 0])
        y_test.append(scaled_test[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    y_test = np.reshape(y_test, (y_test.shape[0],1))
    model = keras.saving.load_model('./models/RNN/RNN_L4_U50_E5_v1_X.keras')
    X_series_pred = scaler.inverse_transform(model.predict(X_test))
    print(f'RNN [X_series_pred]: {X_series_pred.size}')


    dataset_test = Y_series[start:end]
    scaled_test = scaler.fit_transform(dataset_test)
    
    X_test = []
    y_test = []
    for i in range(50, len(scaled_test)):
        X_test.append(scaled_test[i-50:i, 0])
        y_test.append(scaled_test[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    y_test = np.reshape(y_test, (y_test.shape[0],1))
    model = keras.saving.load_model('./models/RNN/RNN_L4_U50_E5_v1_Y.keras')
    Y_series_pred = scaler.inverse_transform(model.predict(X_test))
    print(f'RNN [Y_series_pred]: {Y_series_pred.size}')
    
    dataset_test = Z_series[start:end]
    scaled_test = scaler.fit_transform(dataset_test)
    
    X_test = []
    y_test = []
    for i in range(50, len(scaled_test)):
        X_test.append(scaled_test[i-50:i, 0])
        y_test.append(scaled_test[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    y_test = np.reshape(y_test, (y_test.shape[0],1))
    model = keras.saving.load_model('./models/RNN/RNN_L4_U50_E5_v1_Z.h5')
    Z_series_pred = scaler.inverse_transform(model.predict(X_test))
    print(f'RNN [Z_series_pred]: {Z_series_pred.size}')
    time_period = diff_dataset_from_df['time'].mode()[0]
    
    X_series_predicted_new = (X_series_pred).T
    Y_series_predicted_new = (Y_series_pred).T
    Z_series_predicted_new = (Z_series_pred).T
    
    X_series_pred = X_series_predicted_new[0]
    X_series_pred = X_series_pred.T

    Y_series_pred = Y_series_predicted_new[0]
    Y_series_pred = Y_series_pred.T

    Z_series_pred = Z_series_predicted_new[0]
    Z_series_pred = Z_series_pred.T

    data = {'X': X_series_pred,
            'Y': Y_series_pred,
            'Z': Z_series_pred
            }

    result_df = pd.DataFrame(data)
    result_df = result_df.fillna(0)
    result_df['time'] = time_period
    result_df['trace'] = df['trace'].mode()[0]
    
    result_df_exel_2 = cascade_summ_df(result_df)
    print(f'====RNN cascade result_df_exel_2: ',result_df_exel_2.head())
    
    result_df_exel_2['X'] = result_df_exel_2['X'].apply(lambda n: n + df['X'].iloc[start])
    result_df_exel_2['Y'] = result_df_exel_2['Y'].apply(lambda n: n + df['Y'].iloc[start])
    result_df_exel_2['Z'] = result_df_exel_2['Z'].apply(lambda n: n + df['Z'].iloc[start])
    result_df_exel_2['time'] = result_df_exel_2['time'].apply(lambda n: n + df['time'].iloc[start])
    result_df_exel_2['trace'] = df['trace'].iloc[start]
    result_df_2=result_df_exel_2
    
    print(f'====RNN start time: ',df['time'].iloc[start])

    if(os.path.exists("./models/temp_dataset.xlsx")):
        os.remove("./models/temp_dataset.xlsx")
    result_df_2.to_excel("./models/temp_dataset.xlsx")
    
    return
    
    
if __name__ == '__main__':
    print(__name__)
    main()
