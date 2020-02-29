# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import sklearn libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Import Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting linear regresson model to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting polynomial regresson model to the dataset
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)

