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
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ELM import ELMRegressor

##IMPORT OWN DATASET
forecastData = pd.read_csv (r'C:\Users\Lukas\Documents\Studium\Master\04 SS2020\EI seminar\data\dataset.csv')
#print (forecastData)
#set input and output for algorithm
X = forecastData.iloc[:,1:8].values
y = forecastData.iloc[:,7].values


test_maes_dictionary = dict()

plt.style.use('ggplot')
sns.set_context("talk")
np.random.seed(0)

## DATA PREPROCESSING -> change that to own data set!

diabetes = load_diabetes()
#print(diabetes)
data_filename: diabetes
#X, y = diabetes["data"], diabetes["target"]
#X, y = forecastData["data"], forecastData["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

stdScaler_data = StandardScaler()
X_train = stdScaler_data.fit_transform(X_train)
X_test = stdScaler_data.transform(X_test)

stdScaler_target = StandardScaler()
y_train = stdScaler_target.fit_transform(np.expand_dims(y_train.ravel(),1))  # /max(y_train)
y_test = stdScaler_target.transform(np.expand_dims(y_test,1))  # /max(y_train)
max_y_train = max(abs(y_train.ravel()))
y_train = y_train.ravel() / max_y_train
y_test = y_test / max_y_train

## ELM TRAINING
MAE_TRAIN_MINS = []
MAE_TEST_MINS = []
steps = 10 #new
neurons = 10 #new
for M in range(1, steps , 1):
    MAES_TRAIN = []
    MAES_TEST = []
    # print "Training with %s neurons..."%M
    for i in range(neurons):
        ELM = ELMRegressor(M)
        ELM.fit(X_train, y_train.ravel())
        prediction = ELM.predict(X_train)
        MAES_TRAIN.append(mean_absolute_error(y_train.ravel(),
                                              prediction))

        prediction = ELM.predict(X_test)
        MAES_TEST.append(mean_absolute_error(y_test,
                                             prediction))
    MAE_TEST_MINS.append(min(MAES_TEST))
    MAE_TRAIN_MINS.append(MAES_TRAIN[np.argmin(MAES_TEST)])

print("Minimum MAE ELM =", min(MAE_TEST_MINS))
print ("using amount of steps: ", steps)    #new
print ("using amount of neurons: ", neurons) #new
test_maes_dictionary["ELM"] = min(MAE_TEST_MINS)



#############################################################################################

## PLOTTING THE RESULTS
df = pd.DataFrame()
df["test"] = MAE_TEST_MINS
df["train"] = MAE_TRAIN_MINS

ax = df.plot(figsize=(16, 7))
ax.set_xlabel("Number of Neurons in the hidden layer")
ax.set_ylabel("Mean Absolute Error")
ax.set_title("Extreme Learning Machine error obtained for the Diabetes dataset \n when varying the number of neurons in the "
   "hidden layer (min. at 23 neurons)")
#plt.show()

plt.figure(figsize=(16, 7))
D = test_maes_dictionary
plt.bar(range(len(D)), D.values(), align='center')
plt.xticks(range(len(D)), D.keys())
plt.ylabel("Mean Absolute Error")
plt.title("Error Comparison between Classic Regression Models and ELM")
#plt.show()
