# Data Pre-processing Template - To be imported anywhere

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Import Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

print(dataset.head())
print(X)
print(y)

# Split the dataset to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

print(X_train)
print(y_train)

print(X_test)
print(y_test)

# Feature scaling
"""
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

# Fit the data to linear regressor
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test_set results
y_pred = regressor.predict(X_test)
print(y_pred)
print(y_test)

# Visualising the Training set results
# plt.scatter(X_train, y_train, color='red')
# plt.plot(X_train, regressor.predict(X_train), color='blue')
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
