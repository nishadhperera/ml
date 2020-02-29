# Data Pre-processing Template - To be imported anywhere

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

# Import Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encode the categorical variable to numbers
labelencoder_x = LabelEncoder()
X[:, 3] = labelencoder_x.fit_transform(X[:, 3])

# Get the dummuy variables for the categorical parameter
onehotencoder = OneHotEncoder(handle_unknown='ignore')
enc = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(enc.fit_transform(X), dtype=np.int)

# Avoiding dummy variable trap (automatically done anyway by the library)
X = X[:, 1:]

# Split the dataset to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting the regression model for test set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the results using regressor
y_pred = regressor.predict(X_test)

# Build optimal model using Backward elimination
# Adding 'constant' (intercept) manually as statsmodels does not automatically add it
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

# Remove parameters with a P-value > 0.05 of significance level
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# Automatic parameter elimination
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if regressor_OLS.pvalues[j].astype(float) == maxVar:
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = backwardElimination(X_opt, SL)


# Predict the results with the optimized model
X_opt_test = X_test[:, [0, 3]]
y_pred = regressor_OLS.predict(X_opt_test)
