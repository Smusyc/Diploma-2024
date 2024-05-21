import sys
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from fancyimpute import IterativeSVD
from sklearn.metrics import mean_squared_error

def rolling_window(a, window, intersection=False):
    # result = torch.zeros(size=(a.shape[0],a.shape[1]))
    returns = []
    if intersection:
        for i in range(0, a.shape[0]-window):
            returns.append(a[i:i + window])
    else:
        for i in range(0, a.shape[0],window):
            returns.append(a[i:i + window])
    return np.stack(returns)

class TimeSeriesPreparing:
    def __init__(self):
        pass
    def delete_doubles(self, result_df):
        new_np_trajectory=[]
        for i in range(len(result_df)):
            if(i==0):
                new_np_trajectory.append(result_df[i])
            elif(new_np_trajectory[i-1]!=result_df[i]):
                new_np_trajectory.append(result_df[i])
            
            return pd.DataFrame(copy.deepcopy(new_np_trajectory), columns=['X', 'Y', 'Z', 'time', 'trace'])
            #return new_np_trajectory
    
    
    def fill_missings_by_nans(self, result_df, period):
        new_np_trajectory=[]
        for i in range(len(result_df)):
            if(i==0):
                new_np_trajectory.append(result_df[i])
            elif(new_np_trajectory[i-1][3]==(result_df[i][3]-period)):
                new_np_trajectory.append(result_df[i])
            else:
                new_np_trajectory.append((np.nan, np.nan, np.nan, (new_np_trajectory[i-1][3]+period), result_df[i][4] ))
                new_np_trajectory.append(result_df[i])
        
        
        #return new_np_trajectory
        return pd.DataFrame(copy.deepcopy(new_np_trajectory), columns=['X', 'Y', 'Z', 'time', 'trace'])
        
    def coords_to_vectors(self, result_df):
        new_np_vectors=[]
        for i in range(len(result_df)):
            if(i>0):
                if(result_df[i][0]!=np.nan and result_df[i][1]!=np.nan and result_df[i][2]!=np.nan):
                    new_np_trajectory.append( ( (result_df[i][0]-result_df[i-1][0])/(result_df[i][3]-result_df[i-1][3]), (result_df[i][1]-result_df[i-1][1])/(result_df[i][3]-result_df[i-1][3]), (result_df[i][2]-result_df[i-1][2])/(result_df[i][3]-result_df[i-1][3]), result_df[i-1][3], result_df[i-1][4]) )
                else:
                    new_np_trajectory.append( (np.nan, np.nan, np.nan, result_df[i][3], result_df[i-1][4]))
        return pd.DataFrame(copy.deepcopy(new_np_trajectory), columns=['X', 'Y', 'Z', 'time', 'trace'])
        #return new_np_trajectory
        
    def imputation_timeseries(self, result_df, choice):
    
        X_res_inp = []
        Y_res_inp = []
        Z_res_inp = []
        X_slice = rolling_window(result_df['X'], 100)
        Y_slice = rolling_window(result_df['Y'], 100)
        Z_slice = rolling_window(result_df['Z'], 100)
        
        if(choice=='KNN'):
            imputer = KNNImputer(n_neighbors=2)
            
            X_res_inp = imputer.fit_transform(X_slice).flatten()
            Y_res_inp = imputer.fit_transform(Y_slice).flatten()
            Z_res_inp = imputer.fit_transform(Z_slice).flatten()
        
        elif(choice=='SVD'):
            X_res_inp = IterativeSVD().fit_transform(X_slice).flatten()
            Y_res_inp = IterativeSVD().fit_transform(Y_slice).flatten()
            Z_res_inp = IterativeSVD().fit_transform(Z_slice).flatten()
            

        data = {
        'X': np.array(X_res_inp),
        'Y': np.array(Y_res_inp),
        'Z': np.array(Z_res_inp),
        'time': result_df['time'],
        'trace': result_df['trace']
        }
        
        return pd.DataFrame(data)
