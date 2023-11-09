# -*- coding: utf-8 -*-
"""Final Project Code.ipynb

"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

activity = pd.read_csv("/content/sample_data/dailyActivity_merged.csv")
sleepDay = pd.read_csv("/content/sample_data/sleepDay_merged.csv")

activity = activity.rename(columns = {"ActivityDate": "Date"})
activity['Date'] = pd.to_datetime(activity.Date)
activity['Date'] = activity['Date'].dt.strftime("%m/%d/%y")
activity

sleepDay = sleepDay.rename(columns = {"SleepDay": "Date"})
sleepDay['Date'] = pd.to_datetime(sleepDay.Date)
sleepDay["Date"] = sleepDay['Date'].dt.strftime('%m/%d/%y')
sleepDay

# Integrate two datasets into one data frame
full_data = pd.merge(activity, sleepDay)
full_data = full_data.drop(['Date','TrackerDistance', 'LoggedActivitiesDistance', 'VeryActiveDistance', "ModeratelyActiveDistance", 'LightActiveDistance', 'SedentaryActiveDistance',
                              'VeryActiveMinutes', 'FairlyActiveMinutes', "LightlyActiveMinutes", "SedentaryMinutes"], axis = 1)

# Set target value
full_data["sleepMins"] = full_data["TotalTimeInBed"] - full_data["TotalMinutesAsleep"]
y = full_data["sleepMins"]

# drop unnecessary variables
full_data = full_data.drop(["sleepMins", "TotalSleepRecords", "TotalMinutesAsleep", "TotalTimeInBed"], axis=1)
full_data = full_data.rename(columns = {"Id": "id", "TotalSteps": "steps", "TotalDistance": "distance", "Calories": "calories"})

# re-define the column order
full_data = full_data[['id', 'steps', 'calories', 'distance']]

full_data

# split data into train and test datasets with 80 - 20 size
X_train, X_test, y_train, y_test = train_test_split(full_data, y, test_size=0.2, random_state=42)

X_train.to_csv('X_train_dataset.csv', index=False)

"""# Modeling

# LASSO
"""

lasso = Lasso(alpha = 0.1)
lasso.fit(X_train, y_train)

y_pred_lso = lasso.predict(X_test)

mse_lso = mean_squared_error(y_test, y_pred_lso)

# Print the RMSE value
rmse_lso = np.sqrt(mse_lso)
print("RMSE: ", rmse_lso)

# Using ChatGPT for hyperparameter tuning
lasso = Lasso()

# Define a range of alpha values to search over
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Create a GridSearchCV object with 5-fold cross-validation
grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

best_lasso = grid_search.best_estimator_

y_pred_lso_tuned = best_lasso.predict(X_test)

mse_lso_tuned = mean_squared_error(y_test, y_pred_lso_tuned)

# Print the best hyperparameter values and corresponding mean squared error
print("Best alpha:", grid_search.best_params_)
print("RMSE: ", np.sqrt(mse_lso_tuned))

# Lasso Visualization
plt.scatter(y_test, y_pred_lso_tuned)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.vlines(y_test, y_pred_lso_tuned, y_test, color = 'black')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Comparison of predicted and actual values in Lasso Model')
plt.show()

"""# RIDGE"""

ridge = Ridge(alpha=1.0) # alpha is the regularization parameter
ridge.fit(X_train, y_train)

# Predict on the testing set
y_pred_rdg = ridge.predict(X_test)

mse_rdg = mean_squared_error(y_test, y_pred_rdg)

# Print the RMSE value
rmse_rdg = np.sqrt(mse_rdg)
print("RMSE: ", rmse_rdg)

# Using ChatGPT for hyperparameter tuning
ridge = Ridge()

# Define a range of alpha values to search over
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# Create a GridSearchCV object with 5-fold cross-validation
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

best_rdg = grid_search.best_estimator_

y_pred_rdg_tuned = best_rdg.predict(X_test)

mse_rdg_tuned = mean_squared_error(y_test, y_pred_rdg_tuned)
# Print the best hyperparameter values and corresponding mean squared error
print("Best alpha:", grid_search.best_params_)
print("RMSE: ", np.sqrt(mse_rdg_tuned))

# Ridge Visualization
plt.scatter(y_test, y_pred_rdg)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.vlines(y_test, y_pred_rdg, y_test, color = 'black')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Comparison of predicted and actual values in Ridge Model')
plt.show()

"""# SVM"""

from sklearn.svm import SVC
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Predict on the testing set
y_pred_svm = svm_model.predict(X_test)

# Predict on the test data
mse_svm = mean_squared_error(y_test, y_pred_svm)

# Print the RMSE value
rmse_svm = np.sqrt(mse_svm)
print("RMSE: ", rmse_svm)

#Attempted SVM hyperparameter tuning
# Define the SVM model
svm_model = SVC()

# Define the grid of hyperparameters to search
param_grid = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

# Perform grid search to find the best hyperparameters
svm_grid = GridSearchCV(svm_model, param_grid, cv=5)
svm_grid.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding validation score
print("Best hyperparameters: ", svm_grid.best_params_)
print("Validation score: ", svm_grid.best_score_)

# Train a new SVM model on the full training set using the best hyperparameters
best_svm_model = SVC(**svm_grid.best_params_)
best_svm_model.fit(X_train, y_train)

# SVM Visualization
plt.scatter(y_test, y_pred_svm)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.vlines(y_test, y_pred_svm, y_test, color = 'black')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Comparison of predicted and actual values in SVM Model')
plt.show()

"""# RANDOM FOREST"""

RFmod = RandomForestRegressor(n_estimators = 100, random_state=42).fit(X_train, y_train)
y_pred_rf = RFmod.predict(X_test)
#print(y_pred_rf)

mse_rf = mean_squared_error(y_test, y_pred_rf)

# Print the RMSE value
rmse_rf = np.sqrt(mse_rf)
print("RMSE: ", rmse_rf)

# Using ChatGPT for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 150, 200]
}

RFmod = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(RFmod, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(best_params)

RFmod_tuned = RandomForestRegressor(**best_params, random_state=42).fit(X_train, y_train)
y_pred_rf_tuned = RFmod_tuned.predict(X_test)

mse_rf_tuned = mean_squared_error(y_test, y_pred_rf_tuned)

# Print the RMSE value
rmse_rf_tuned = np.sqrt(mse_rf_tuned)
print("RMSE: ", rmse_rf_tuned)

# Random Forest Visualization
plt.scatter(y_test, y_pred_rf_tuned)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.vlines(y_test, y_pred_rf, y_test, color = 'black')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Comparison of predicted and actual values in Random Forest Model')
plt.show()

"""# ENSEMBLE"""

# Performing ensemble on the best performing models above
from sklearn.ensemble import VotingRegressor

estimators = [('lasso', best_lasso), ('random_forest', RFmod_tuned)]
ensemble = VotingRegressor(estimators)
ensemble.fit(X_train, y_train)

# Predict on the testing set
y_pred_ensemble = ensemble.predict(X_test)

mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)

# Print the RMSE value
rmse_ensemble = np.sqrt(mse_ensemble)
print("RMSE: ", rmse_ensemble)

# Ensemble Visualization
plt.scatter(y_test, y_pred_ensemble)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.vlines(y_test, y_pred_ensemble, y_test, color = 'black')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Comparison of predicted and actual values in Ensemble Model')
plt.show()