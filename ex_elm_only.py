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
forecastData = pd.read_csv('data\merged_temperature.csv', parse_dates=["MESS_DATUM"], infer_datetime_format=True,
                           index_col="MESS_DATUM")


# set input and output for algorithm
##change dataset to supervised problem
def rmse(pred, y):
    return np.sqrt(np.mean((pred - y)**2))


def to_supervised(series):
    cols = []
    targets = []
    for i in range(24):
        cols.append((series.shift(i)))
    for i in range(24, 25):
        targets.append((series.shift(i)))
    return np.stack(cols, 1)[25:, :], np.stack(targets, 1)[25:, :]

bldg_data  = forecastData["Bldg.124"]
stdScaler_data = StandardScaler()

#bldg_data = stdScaler_data.fit_transform(bldg_data.values.reshape((-1,1)))
#X, y = to_supervised(pd.Series(bldg_data.reshape((-1,))))
X, y = to_supervised(bldg_data)

test_maes_dictionary = dict()

plt.style.use('ggplot')
sns.set_context("talk")
np.random.seed(0)

## DATA PREPROCESSING


X_train, X_test = X[:25000], X[25000:]
y_train, y_test = y[:25000], y[25000:]

# X_train = stdScaler_data.fit_transform(X_train)
# X_test = stdScaler_data.transform(X_test)

# y_train = stdScaler_target.fit_transform(np.expand_dims(y_train.ravel(),1))  # /max(y_train)
# y_test = stdScaler_target.transform(np.expand_dims(y_test,1))  # /max(y_train)
max_y_train = max(abs(y_train.ravel()))

## ELM TRAINING
MAE_TRAIN_MINS = []
MAE_TEST_MINS = []
steps = 2  # new
neurons = 10  # new
predictions = []
for M in range(1, steps, 1):
    MAES_TRAIN = []
    MAES_TEST = []
    # print "Training with %s neurons..."%M
    for i in [10, 100, 300, 500]:
        print(f"Training {i} neurons in Step{M}")
        ELM = ELMRegressor(i)
        ELM.fit(X_train, y_train)
        prediction = ELM.predict(X_train)
        MAES_TRAIN.append(mean_absolute_error(y_train,
                                              prediction))

        prediction = ELM.predict(X_test)
        predictions.append(prediction)
        mae = mean_absolute_error(y_test,
                                             prediction)
        MAES_TEST.append(mae)
        print(f"MAE: {mae}")
        print(f"RMSE: {rmse(prediction, y_test)}")
        print(f"MSE: {mean_squared_error(prediction, y_test)}")
    MAE_TEST_MINS.append(min(MAES_TEST))
    MAE_TRAIN_MINS.append(MAES_TRAIN[np.argmin(MAES_TEST)])

print("Minimum MAE ELM =", min(MAE_TEST_MINS))
print("using amount of steps: ", steps)  # new
print("using amount of neurons: ", neurons)  # new
test_maes_dictionary["ELM"] = min(MAE_TEST_MINS)

for prediction in predictions:
    p = pd.DataFrame(prediction.clip(0,15), index=forecastData.index[25 + len(y_train):])
    bldg = pd.DataFrame(y_test.reshape((-1,)), index=forecastData.index[25 + len(y_train):])
   # p = pd.DataFrame(prediction[:,1], index=forecastData.index[25 + len(y_train):])
    p.plot()
    #bldg.plot()
    plt.show()

# bldg = pd.DataFrame(y_test.reshape((-1,)), index=forecastData.index[25 + len(y_train):])

#bldg.plot()
#plt.show()

#############################################################################################

## PLOTTING THE RESULTS
# df = pd.DataFrame()
# df["test"] = MAE_TEST_MINS
# df["train"] = MAE_TRAIN_MINS
#
# ax = df.plot(figsize=(16, 7))
# ax.set_xlabel("Number of Neurons in the hidden layer")
# ax.set_ylabel("Mean Absolute Error")
# ax.set_title("Extreme Learning Machine error obtained for the Diabetes dataset \n when varying the number of neurons in the "
#    "hidden layer (min. at 23 neurons)")
# #plt.show()
#
# plt.figure(figsize=(16, 7))
# D = test_maes_dictionary
# plt.bar(range(len(D)), D.values(), align='center')
# plt.xticks(range(len(D)), D.keys())
# plt.ylabel("Mean Absolute Error")
# plt.title("Error Comparison between Classic Regression Models and ELM")
# plt.show()
