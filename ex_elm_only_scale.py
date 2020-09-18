# -*- coding: utf-8 -*-
__author__ = 'ivanvallesperez'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from ELM import ELMRegressor

##IMPORT OWN DATASET
forecastData = pd.read_csv(r'C:\Users\Lukas\Documents\Studium\Master\04 SS2020\EI seminar\data\merged_temperature.csv', parse_dates=["MESS_DATUM"], infer_datetime_format=True,
                           index_col="MESS_DATUM")


# set input and output for algorithm
##change dataset to supervised problem with 24 hours watch 1 hour forecast 
#Root mean squared error
def rmse(pred, y):
    return np.sqrt(np.mean((pred - y)**2))


#Root mean squared error clipped
def rmse_clip(pred, y):
    return np.sqrt(np.mean((pred - y)**2)).clip(0,100)

hoursLookedAt = 24
hoursToPredict = hoursLookedAt + 1


# Scale 1
bldg_data  = forecastData["Bldg.124"]
stdScaler_data = StandardScaler()
SCALING=True
if SCALING:
    bldg_data = pd.Series(stdScaler_data.fit_transform(bldg_data.values.reshape((-1,1))).reshape((-1,)))

def to_supervised(series):
    cols = []
    targets = []
    for i in range(hoursLookedAt):
        cols.append((series.shift(-i)))
    for i in range(hoursLookedAt, hoursToPredict):
        targets.append((series.shift(-i)))
    return np.stack(cols, 1)[:-hoursToPredict:, :], np.stack(targets, 1)[:-hoursToPredict:, :]




X, y = to_supervised(bldg_data)

test_maes_dictionary = dict()

plt.style.use('ggplot') 
sns.set_context("talk")
np.random.seed(0)

## DATA PREPROCESSING

X_train, X_test = X[31068:57347], X[57347:] 
y_train, y_test = y[31068:57347], y[57347:]

# Scale 2
if SCALING:
    y_test = stdScaler_data.inverse_transform(y_test)


max_y_train = max(abs(y_train.ravel()))

## ELM TRAINING
MAE_TRAIN_MINS = []
MAE_TEST_MINS = []
steps = 3  # new
predictions = []
print("Forecast horizon in hours: ", hoursToPredict - hoursLookedAt)
for M in range(1, steps, 1):
    MAES_TRAIN = []
    MAES_TEST = []
    # print "Training with %s neurons..."%M
    for i in [100]:
        print(f"Training {i} neurons in Step {M}")
        ELM = ELMRegressor(i)
        ELM.fit(X_train, y_train)
        prediction = ELM.predict(X_train)
    
        MAES_TRAIN.append(mean_absolute_error(y_train, 
                                              prediction))
        
        prediction = ELM.predict(X_test)
      
        #Scale 3
        if SCALING:
            prediction = stdScaler_data.inverse_transform(prediction)
        
        predictions.append(prediction)
        mae = mean_absolute_error(y_test,
                                             prediction)


        MAES_TEST.append(mae)
        
        print(f"RMSE: {rmse(prediction, y_test)}")
        print(f"RMSE: {rmse_clip(prediction, y_test)}")
        #print(f"MAE: {mae}")
        #print(f"MSE: {mean_squared_error(prediction, y_test)}")
    
    MAE_TEST_MINS.append(min(MAES_TEST))
    MAE_TRAIN_MINS.append(MAES_TRAIN[np.argmin(MAES_TEST)])

print("Minimum MAE ELM =", min(MAE_TEST_MINS))
print("using amount of steps: ", steps)  # new

#print("using amount of neurons: ", neurons)  # new
test_maes_dictionary["ELM"] = min(MAE_TEST_MINS)


##Showing results

for prediction in predictions:
    ##Für time indices
    ##clipping schneidet Ausreißer ab
    p = pd.DataFrame(prediction.clip(0,15)[:,0], index=forecastData.index[31068+ hoursToPredict + len(y_train):])
    bldg = pd.DataFrame(y_test[:,0].reshape((-1,)), index=forecastData.index[31068+ hoursToPredict + len(y_train):])
    
    ##show results    
    ax = p.plot()
    bldg.plot(ax=ax, alpha=0.2)

  

plt.show()

