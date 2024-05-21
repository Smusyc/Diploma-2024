import sys
import pandas as pd
import numpy as np
import os
import copy
import math
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
#print(__name__)


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
    #print(f"for {index} in range({(new_df.index)[0]})")
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
    #new_df_2.drop(index=indexes[0], axis= 0 , inplace= True )
    return new_df_2


def main():
    print('=======================================================')
    start = int(sys.argv[2])

    
    end = int(sys.argv[3])

    
    df = pd.read_excel(sys.argv[1])
    #df_for_training = pd.read_excel("./train_data/ARIMA/dataset_for_ARIMA_trainer.xlsx")
    result_df_2 = pd.read_excel("./models/MyNN/MyNNResult.xlsx")
    
    
    #print(f'====AUTOATIMA start time: ',df['time'].iloc[start])
    
    ##Убрать потом три строки коментария ВЕРНУТЬ
    if(os.path.exists("./models/temp_dataset.xlsx")):
        os.remove("./models/temp_dataset.xlsx")
    result_df_2.to_excel("./models/temp_dataset.xlsx")
    
    #print('here6')
    #result_df = result_df.to_records(index=False)
    #return (result_df.tostring(encoding='utf-8')).tobytes()
    return
    
    
if __name__ == '__main__':
    print(__name__)
    #ВЕРНУТЬ
    main()