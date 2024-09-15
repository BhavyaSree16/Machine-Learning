"""Import Libraries"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error

"""Reading the csv File"""

df=pd.read_csv('ToyotaCorolla.csv')

df.keys()

# df.info()

""" Selecting the columns"""

df = df[["Price", "Age_08_04", "KM", "HP", "cc", "Doors", "Gears", "Quarterly_Tax", "Weight"]]

df.shape

df.head()

print(df["Price"])

print(df['Price'].max())
print(df['Price'].min())

import seaborn as sns
corr_results = df.corr()
fig = plt.figure(figsize = (12,7))

sns.heatmap(corr_results,annot = True)
plt.title('Co-relation Matrix')
plt.show()

# df = df[["Price", "KM", "HP", "cc", "Doors", "Gears", "Quarterly_Tax", "Weight"]]

# import matplotlib.pyplot as plt

# # Create box plots for each column
# df.plot(kind='box', subplots=True, layout=(3,3), figsize=(12,9))

# # Adjust layout and display the plots
# plt.tight_layout()
# plt.show()

df.head()

"""Splitting the data into features (X) and target (y)"""

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

X.shape

len(X)

y

""" Splitting the data into training and testing sets"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

print("Shape of X_train: ",X_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test:",y_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# from sklearn.preprocessing import MinMaxScaler

# # Initialize the scaler
# scaler = MinMaxScaler()

# # Normalize all columns
# df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# # Print the normalized DataFrame
# print(df_normalized.head())

"""### Linear Regression"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test), 1)), axis = 1)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Root Mean Squared Error
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# R-squared
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R²): {r2:.2f}")

# to know Accuracy
mse = mean_squared_error(y_test, y_pred)
accuracy = 1 - mse / y_test.var()
print('Accuracy: ', accuracy)

def calculate_mse_r2(y_test, y_predict):
    # Convert inputs to numpy arrays for easier manipulation
    y_test = np.array(y_test)
    y_predict = np.array(y_predict).astype(int)
    #print(y_test)
    #print(y_predict)
    l = len(y_test)
    sum = 0;
    for i in range(l):
        print(y_test[i]," ",y_predict[i]," ",y_test[i]-y_predict[i])
        sum += (y_test[i] - y_predict[i]) ** 2
    print(sum)
    print(l)
    mse = sum / l

    # # Calculate MSE (Mean Squared Error)
    # mse = np.mean((y_test - y_predict) ** 2)

    # Calculate R-squared (Coefficient of Determination)
    ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
    ss_residual = np.sum((y_test - y_predict) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    return mse, r2

mse, r2 = calculate_mse_r2(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")