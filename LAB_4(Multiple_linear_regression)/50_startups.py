"""Import Libraries"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

"""Reading Data"""

df=pd.read_csv('50_Startups.csv')

df.info()

df.boxplot()

"""Dropping non Contributing Features"""

X=df.drop('State', axis=1)
X

"""Preprocessing The whole dataset
*   Use Minmax Scaler
"""

from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler().fit(X)
X = scaler.transform(X)
print(X)

"""Data Split"""

y = df['Profit']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""Model Fitting and Pipelining"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

"""Predicting and Evaluation"""

y_pred = regressor.predict(X_test)

#R2_score
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))

#RMSE
from sklearn.metrics import mean_squared_error
print(np.sqrt(mean_squared_error(y_test,y_pred)))

#MSE
print(mean_squared_error(y_test,y_pred))