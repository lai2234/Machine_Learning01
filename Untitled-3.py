#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """@author: TsunPang Lai"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score

# Importing the dataset
dataset = pd.read_csv('/Users/Pang/Downloads/RedditData.csv')

# Creating a dataset of predictors and target
X= dataset.iloc [:, 0:8].values #predictor dataset IV
y= dataset.iloc [:, 8:9].values #target dataset DV

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, 
random_state = 0)

from sklearn.metrics import explained_variance_score, mean_absolute_error



# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor

# Creating an instance of Decision Tree Regression
regressor_dt= DecisionTreeRegressor(random_state= 0, max_depth=3 )
regressor_dt.fit(X_train, y_train)
y_pred_dt= regressor_dt.predict(X_test)

# Printing the metrics of the model
print('Variance score: %.2f', explained_variance_score(y_test, y_pred_dt))
print('MAE score: %.2f', mean_absolute_error(y_test, y_pred_dt))



# Fitting Multiple Linear Regression to the Training set and reporting the accuracy
from sklearn.linear_model import LinearRegression

# Creating an instance of Linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred=regressor.predict(X_test)

# Printing the metrics of the model
print('Variance score: %.2f', explained_variance_score(y_test, y_pred))
print('MAE score: %.2f', mean_absolute_error(y_test, y_pred))



# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

# Creating an instance of random forest regressor
rf_regressor = RandomForestRegressor(n_estimators=5, random_state=0)
rf_regressor.fit(X,y)

y_pred_rf = rf_regressor.predict(X_test)
print('Variance score: %.2f', explained_variance_score(y_test, y_pred_rf))
print('MAE score: %.2f',mean_absolute_error(y_test, y_pred_rf))



