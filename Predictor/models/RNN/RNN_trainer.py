import pandas as pd
import numpy as np
import os
import math
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import copy

# importing libraries
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.optimizers import SGD
#from sklearn import metrics
#from sklearn.metrics import mean_squared_error


def difference_df(df):
    new_df = df.rolling(2).apply(lambda n: (np.array(n))[1] - (np.array(n))[0])
    new_df = new_df.fillna(0)
    return new_df

def cascade_summ_df(new_df):
    new_df_2 = copy.deepcopy(new_df)
    new_df_index = new_df.index
    accum_row = new_df.loc[new_df_index[0]]
    for index, row in new_df.iterrows():
        accum_row += row
        new_df_2.loc[index] += accum_row
        
    return new_df_2

#def main(data, mode, start = 0, end = 0, ):
def main():

    start = int(sys.argv[2])
    print(f'AUTOATIMA [start]: {sys.argv[2]}')
    
    end = int(sys.argv[3])
    print(f'AUTOATIMA [end]: {sys.argv[3]}')
    
    df = pd.read_excel(sys.argv[1])
    epochs_number=5
    end = len(data)
    training_data_len = math.ceil(len(data) * (start/end) )
    training_data_len

    #Splitting the dataset

    train_data = data[:training_data_len]
    test_data = data[training_data_len:]

    dataset_train = train_data


    # Reshaping 1D to 2D array
    dataset_train = np.reshape(dataset_train, (-1,1))
    dataset_train.shape

    scaler = MinMaxScaler(feature_range=(0,1))
    # scaling dataset
    scaled_train = scaler.fit_transform(dataset_train)

    print(scaled_train[:5])

    #Для тестовых данных
    # Selecting Open Price values
    #dataset_test = test_data.Open.values
    dataset_test = test_data

    #Reshaping 1D to 2D array
    dataset_test = np.reshape(dataset_test, (-1,1))
    # Normalizing values between 0 and 1
    scaled_test = scaler.fit_transform(dataset_test)
    print(*scaled_test[:5])


    X_train = []
    y_train = []
    for i in range(50, len(scaled_train)):
        X_train.append(scaled_train[i-50:i, 0])
        y_train.append(scaled_train[i, 0])
        if i <= 51:
            print(X_train)
            print(y_train)
            print()

    X_test = []
    y_test = []
    for i in range(50, len(scaled_test)):
        X_test.append(scaled_test[i-50:i, 0])
        y_test.append(scaled_test[i, 0])



    # The data is converted to Numpy array
    X_train, y_train = np.array(X_train), np.array(y_train)

    #Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
    y_train = np.reshape(y_train, (y_train.shape[0],1))
    print("X_train :",X_train.shape,"y_train :",y_train.shape)

    # The data is converted to numpy array
    X_test, y_test = np.array(X_test), np.array(y_test)

    #Reshaping
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
    y_test = np.reshape(y_test, (y_test.shape[0],1))
    print("X_test :",X_test.shape,"y_test :",y_test.shape)




    # initializing the RNN
    regressor = Sequential()
    # adding RNN layers and dropout regularization
    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True,
                            input_shape = (X_train.shape[1],1)))
    regressor.add(Dropout(0.2))
    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True))

    regressor.add(SimpleRNN(units = 50,
                            activation = "tanh",
                            return_sequences = True))

    regressor.add( SimpleRNN(units = 50))

    # adding the output layer
    regressor.add(Dense(units = 1,activation='sigmoid'))

    # compiling RNN
    '''
    #Old style
    epochs = 50
    learning_rate = 0.01
    decay_rate = learning_rate / epochs
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate, decay=decay_rate)
    '''

    '''
    #New style
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=0.01,
      decay_steps=10000,
      decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    '''
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=0.01,
      decay_steps=10000,
      decay_rate=0.9)

    '''
    #Old Version
    regressor.compile(optimizer = SGD(learning_rate=0.01,
                                      decay=1e-6,
                                      momentum=0.9,
                                      nesterov=True),
                      loss = "mean_squared_error")
    '''

    regressor.compile(optimizer = SGD(learning_rate=lr_schedule,
                                      momentum=0.9,
                                      nesterov=True),
                      loss = "mean_squared_error")

    # fitting the model
    regressor.fit(X_train, y_train, epochs = epochs_number, batch_size = 2)
    regressor.summary()
    result_regressor = regressor
    #prediction = scaler.inverse_transform(result_regressor.predict(X_test))

    return prediction