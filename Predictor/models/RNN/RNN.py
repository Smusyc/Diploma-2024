'''
print('=======================================================')
    start = int(sys.argv[2])
    print(f'RNN [start]: {sys.argv[2]}')
    
    end = int(sys.argv[3])
    print(f'RNN [end]: {sys.argv[3]}')
    
    df = pd.read_excel(sys.argv[1])
    #df_for_training = pd.read_excel("./train_data/ARIMA/dataset_for_ARIMA_trainer.xlsx")
    #df_for_training = pd.read_excel("./train_data/ARIMA/dataset_for_ARIMA_trainer.xlsx")
    #df_for_training = pd.read_excel(sys.argv[1])
    print(f'RNN [dataset]: {sys.argv[1]}')
    

    print(f'RNN [X_series]: ',df['X'][start:end].head())
    print(f'RNN [Y_series]: ',df['Y'][start:end].head())
    print(f'RNN [Z_series]: ',df['Z'][start:end].head())
    
    diff_dataset_from_df = difference_df(df)
    #print(f'RNN diff [X_series]: ',diff_dataset_from_df['X'][start:end].head())
    #print(f'RNN diff [Y_series]: ',diff_dataset_from_df['Y'][start:end].head())
    #print(f'RNN diff [Z_series]: ',diff_dataset_from_df['Z'][start:end].head())
    #diff_dataset_for_training = difference_df(df_for_training)
    #diff_dataset_for_training = diff_dataset_for_training.drop(['Unnamed: 0'], axis=1)
    #diff_dataset_for_training.to_excel("./train_data/ARIMA/diff_dataset_for_training.xlsx")
    #print(f'AUTOARIMA diff_dataset_for_training [columns]: {diff_dataset_for_training.columns}')
    
    X_series = diff_dataset_from_df['X']
    #print(f'RNN [X_series_difference]: {X_series.head()}; Index = {(X_series.index)[0]}')
    
    Y_series = diff_dataset_from_df['Y']
    #print(f'RNN [Y_series]: {Y_series.head()}; Index = {(Y_series.index)[0]}')
    
    Z_series = diff_dataset_from_df['Z']
    #print(f'RNN [Z_series]: {Z_series.head()}; Index = {(Z_series.index)[0]}')
'''

import tensorflow as tf
import sys
import pandas as pd
import numpy as np
import os
import copy
import math
import keras

from sklearn.preprocessing import MinMaxScaler
#import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import GRU, Bidirectional
from keras.optimizers import SGD
from sklearn import metrics
from sklearn.metrics import mean_squared_error
#print(__name__)


def difference_df(df):
    '''
    new_df = df.rolling(2).apply(lambda n: (np.array(n))[1] - (np.array(n))[0])
    new_df = new_df.fillna(0)
    indexes = new_df.index
    new_df.drop (index=indexes[0], axis= 0 , inplace= True )
    return new_df
    '''
    '''
    df = df.apply(lambda x: np.square(x) if x.name == 'b' else x, 
                  axis = 1)
    '''
    new_df = pd.DataFrame()

    new_df['X'] = (df['X'].rolling(2)).apply(lambda n: (np.array(n))[1] - (np.array(n))[0])
    new_df['Y'] = (df['Y'].rolling(2)).apply(lambda n: (np.array(n))[1] - (np.array(n))[0])
    new_df['Z'] = (df['Z'].rolling(2)).apply(lambda n: (np.array(n))[1] - (np.array(n))[0])
    new_df['time'] = (df['time'].rolling(2)).apply(lambda n: (np.array(n))[1] - (np.array(n))[0])
    new_df['trace'] = (df['trace'].rolling(2)).apply(lambda n: (np.array(n))[1])
    new_df = new_df.fillna(0)
    print(f"indexes = {(new_df.index)[-1]}")
    for index in range(1, (new_df.index)[-1]):
    #print(f"for {index} in range({(new_df.index)[0]})")
        if (new_df['trace'].iloc[index] != new_df['trace'].iloc[index+1]):
            #print(" Before")
            #print(f"new_df[{index}] trace = {new_df['trace'].iloc[index]}; X[{new_df['Z'].iloc[index]}]; Y[{new_df['Y'].iloc[index]}]; Z[{new_df['Z'].iloc[index]}];")
            #print(f"new_df[{index+1}] trace = {new_df['trace'].iloc[index+1]}; X[{new_df['X'].iloc[index+1]}]; Y[{new_df['Y'].iloc[index+1]}]; Z[{new_df['Z'].iloc[index+1]}];")
            new_df['X'].iloc[index+1] = new_df['X'].iloc[index]
            new_df['Y'].iloc[index+1] = new_df['Y'].iloc[index]
            new_df['Z'].iloc[index+1] = new_df['Z'].iloc[index]
            new_df['time'].iloc[index+1] = new_df['time'].iloc[index]
            #print(" After")
            #print(f"new_df[{index}] trace = {new_df['trace'].iloc[index]}; X[{new_df['Z'].iloc[index]}]; Y[{new_df['Y'].iloc[index]}]; Z[{new_df['Z'].iloc[index]}];")
            #print(f"new_df[{index+1}] trace = {new_df['trace'].iloc[index+1]}; X[{new_df['X'].iloc[index+1]}]; Y[{new_df['Y'].iloc[index+1]}]; Z[{new_df['Z'].iloc[index+1]}];")
            #new_df = new_df.drop(index=index )
          

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
    print('=======================================================')
    start = int(sys.argv[2])
    #print(f'AUTOATIMA [start]: {sys.argv[2]}')
    
    end = int(sys.argv[3])
    #print(f'AUTOATIMA [end]: {sys.argv[3]}')
    
    df = pd.read_excel(sys.argv[1])
    df_for_training = df
    #df_for_training = pd.read_excel("./train_data/ARIMA/dataset_for_ARIMA_trainer.xlsx")
    #(pd.concat([df_for_training,df], ignore_index = True, axis=0)).to_excel("./models/check_result_concat.xlsx") 
    #df_for_training = pd.read_excel(sys.argv[1])
    #print(f'AUTOATIMA [dataset]: {sys.argv[1]}')
    

    #print(f'AUTOATIMA [X_series]: ',df['X'][start:end].head())
    #print(f'AUTOATIMA [Y_series]: ',df['Y'][start:end].head())
    #print(f'AUTOATIMA [Z_series]: ',df['Z'][start:end].head())
    
    diff_dataset_from_df = difference_df(df)
    diff_dataset_for_training = difference_df(df_for_training)
    #diff_dataset_for_training = diff_dataset_for_training.drop(['Unnamed: 0'], axis=1)
    #diff_dataset_for_training.to_excel("./train_data/ARIMA/diff_dataset_for_training.xlsx")
    #print(f'AUTOARIMA diff_dataset_for_training [columns]: {diff_dataset_for_training.columns}')
    
    X_series = diff_dataset_for_training['X']
    #print(f'AUTOATIMA [X_series_difference]: {X_series.head()}; Index = {(X_series.index)[0]}')
    
    Y_series = diff_dataset_for_training['Y']
    #print(f'AUTOATIMA [Y_series]: {Y_series.head()}; Index = {(Y_series.index)[0]}')
    
    Z_series = diff_dataset_for_training['Z']
    #print(f'AUTOATIMA [Z_series]: {Z_series.head()}; Index = {(Z_series.index)[0]}')
    
    #time = (df['time'])[start:end]
    #print(f'AUTOATIMA [time]: {time.head()}; Index = {(time.index)[0]}')
    
    #trace = (df['trace'])[start:end]
    #print(f'AUTOATIMA [trace]: {trace.head()}; Index = {(trace.index)[0]}')
    scaler = MinMaxScaler(feature_range=(0,1))
    
    
    
    dataset_test = X_series[start:end]#Тут менять
    print('dataset_test')
    print(dataset_test)
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
    #model = ARIMA(X_series, order=(2,1,0))
    #model = keras.saving.load_model('./models/RNN/RNN_L4_U50_E5_v1_X.keras')
    model = keras.saving.load_model('./models/RNN/RNN_L4_U50_E5_v1_X.h5')
    X_series_pred = scaler.inverse_transform(model.predict(X_test))
    #X_series_pred = scaler.inverse_transform(model.predict(X_series[start:end]))
    #model = ARIMA(X_series)
    #X_series_pred = model.fit().predict(start=start, end=end-1)
    print(f'RNN [X_series_pred]: {X_series_pred.size}')


    dataset_test = Y_series[start:end]#Тут менять
    #dataset_test = np.reshape(dataset_test, (-1,1))
    scaled_test = scaler.fit_transform(dataset_test)
    
    X_test = []
    y_test = []
    for i in range(50, len(scaled_test)):
        X_test.append(scaled_test[i-50:i, 0])
        y_test.append(scaled_test[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    y_test = np.reshape(y_test, (y_test.shape[0],1))
    #model = keras.saving.load_model('./models/RNN/RNN_L4_U50_E5_v1_Y.keras')
    model = keras.saving.load_model('./models/RNN/RNN_L4_U50_E5_v1_Y.h5')
    Y_series_pred = scaler.inverse_transform(model.predict(X_test))
    #Y_series_pred = model.predict(start=start, end=end-1)
    #model = ARIMA(Y_series, order=(2,1,0))
    #model = ARIMA(Y_series)
    #Y_series_pred = model.fit().predict(start=start, end=end-1)
    print(f'RNN [Y_series_pred]: {Y_series_pred.size}')
    
    dataset_test = Z_series[start:end]#Тут менять
    #dataset_test = np.reshape(dataset_test, (-1,1))
    scaled_test = scaler.fit_transform(dataset_test)
    
    X_test = []
    y_test = []
    for i in range(50, len(scaled_test)):
        X_test.append(scaled_test[i-50:i, 0])
        y_test.append(scaled_test[i, 0])
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    y_test = np.reshape(y_test, (y_test.shape[0],1))
    #model = keras.saving.load_model('./models/RNN/RNN_L4_U50_E5_v1_Z.keras')
    model = keras.saving.load_model('./models/RNN/RNN_L4_U50_E5_v1_Z.h5')
    Z_series_pred = scaler.inverse_transform(model.predict(X_test))
    #Z_series_pred = model.predict(start=start, end=end-1)
    #model = ARIMA(Z_series, order=(2,1,0))
    #model = ARIMA(Z_series)
    #Z_series_pred = model.fit().predict(start=start, end=end-1)
    print(f'RNN [Z_series_pred]: {Z_series_pred.size}')
    
    #model = ARIMA(dataset['time'], order=(2,1,0))
    #time_pred = model.fit().predict(start=start, end=end)
    #time_pred = time_pred.apply(lambda x: int(x))
    #print(f'AUTOATIMA [Z_series_pred]: {time_pred.tail()}')
    
    #model = ARIMA(dataset['trace'], order=(2,1,0))
    #trace_pred = model.fit().predict(start=start, end=end)
    #print(f'AUTOATIMA [Z_series_pred]: {trace_pred.tail()}')
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
    #print(f'====AUTOATIMA dataset: ',dataset.head())
    #print(f'====AUTOATIMA cascade result_df: ',result_df.head())
    
    result_df_exel_2 = cascade_summ_df(result_df)
    print(f'====RNN cascade result_df_exel_2: ',result_df_exel_2.head())
    
    result_df_exel_2['X'] = result_df_exel_2['X'].apply(lambda n: n + df['X'].iloc[start])
    result_df_exel_2['Y'] = result_df_exel_2['Y'].apply(lambda n: n + df['Y'].iloc[start])
    result_df_exel_2['Z'] = result_df_exel_2['Z'].apply(lambda n: n + df['Z'].iloc[start])
    result_df_exel_2['time'] = result_df_exel_2['time'].apply(lambda n: n + df['time'].iloc[start])
    #result_df_exel_2['trace'] = df['trace'].iloc[start]
    result_df_exel_2['trace'] = df['trace'].iloc[start]
    result_df_2=result_df_exel_2
    
    print(f'====RNN start time: ',df['time'].iloc[start])
    
    ##Убрать потом три строки коментария ВЕРНУТЬ
    if(os.path.exists("./models/temp_dataset.xlsx")):
        os.remove("./models/temp_dataset.xlsx")
    result_df_2.to_excel("./models/temp_dataset.xlsx")
    
    print('here6')
    #result_df = result_df.to_records(index=False)
    #return (result_df.tostring(encoding='utf-8')).tobytes()
    return
    
    
if __name__ == '__main__':
    print(__name__)
    #ВЕРНУТЬ
    main()