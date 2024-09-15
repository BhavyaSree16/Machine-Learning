"""Import Libraries"""
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
df = pd.DataFrame(data=housing['data'])
X = housing.data
X = np.c_[np.ones((len(X), 1)), X]
Y = housing.target

theta_normal = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)

print(theta_normal)

Y_pred = X.dot(theta_normal)

print(Y_pred)

"""Calculating Residual - SSE"""

sum = 0
for i in range(Y_pred.shape[0]):
  sum+= (Y[i]-Y_pred[i])**2
print(sum)

#R2_Score from Scratch
ss_total = np.sum((Y - np.mean(Y)) ** 2)
ss_residual = np.sum((Y - Y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)
print(r2)

"""Implementing Gradient Descent - Full batch"""

from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
df = pd.DataFrame(data=housing['data'])
x = housing.data
x = np.c_[np.ones((len(X), 1)), x]
y = housing.target
model = LinearRegression()
model.fit(X,Y)

y_pred = model.predict(x)

print(model.intercept_,model.coef_[0])

#SSE from Scratch for Full batch gradient
sum = 0
for i in range(y_pred.shape[0]):
  sum+= (y[i]-y_pred[i])**2
print(sum)

#R2_Score from Scratch
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_residual = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)
print(r2)

"""Implementing using Stochastic Gradient Descent"""

from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
df = pd.DataFrame(data=housing['data'])
x = housing.data
x = np.c_[np.ones((len(X), 1)), x]
y = housing.target
model = LinearRegression()
model.fit(X,Y)

y_pred = model.predict(x)

print(model.intercept_,model.coef_[0])

#SSE from Scratch for Full batch gradient
sum = 0
for i in range(y_pred.shape[0]):
  sum+= (y[i]-y_pred[i])**2
print(sum)

#R2_Score from Scratch
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_residual = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)
print(r2)

"""Implementing using Stochastic Gradient Descent"""

from sklearn.linear_model import SGDRegressor
model = SGDRegressor()
model.fit(x,y)

y_pred = model.predict(x)

#SSE from Scratch for Stochastic batch gradient
sum = 0
for i in range(y_pred.shape[0]):
  sum+= (y[i]-y_pred[i])**2
print(sum)

#R2_Score from Scratch for Stochastic Gradient Descent
ss_total = np.sum((y - np.mean(y)) ** 2)
ss_residual = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)
print(r2)

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()
X = housing.data
Y = housing.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
Y_pred_batch_gd = lin_reg.predict(X_test)

sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd_reg.fit(X_train, Y_train)
Y_pred_sgd = sgd_reg.predict(X_test)

mse_batch_gd = mean_squared_error(Y_test, Y_pred_batch_gd)
mse_sgd = mean_squared_error(Y_test, Y_pred_sgd)
r2_batch_gd = r2_score(Y_test, Y_pred_batch_gd)
r2_sgd = r2_score(Y_test, Y_pred_sgd)

print("Full-Batch Gradient Descent Mean Squared Error:", mse_batch_gd)
print("Stochastic Gradient Descent Mean Squared Error:", mse_sgd)
print("Full-Batch Gradient Descent R² Score:", r2_batch_gd)
print("Stochastic Gradient Descent R² Score:", r2_sgd)