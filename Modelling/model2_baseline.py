#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression

def main(data_train, data_test, data_val):
    # Filter the data
    data_train = data_train[data_train['meal'] == 1]
    data_test = data_test[data_test['meal'] == 1]
    data_val = data_val[data_val['meal'] == 1]

    # Adjust the 'CHO_total' values
    data_train['CHO_total'] = data_train['CHO_total'] * 3
    data_test['CHO_total'] = data_test['CHO_total'] * 3
    data_val['CHO_total'] = data_val['CHO_total'] * 3

    # Define the features and the target variable
    X_train = data_train.drop([ 'subject','meal', 'start_block', 'end_block','CHO_total'], axis=1)
    Y_train = data_train['CHO_total']
    X_test = data_test.drop(['subject','meal', 'start_block', 'end_block','CHO_total'], axis=1)
    Y_test = data_test['CHO_total']
    X_val = data_val.drop([ 'subject','meal', 'start_block', 'end_block','CHO_total'], axis=1)
    Y_val = data_val['CHO_total']

    # Create linear regression object
    regressor = LinearRegression()

    # Train the model using the training set
    regressor.fit(X_train, Y_train)

    # Make predictions using the training, validation and testing set
    Y_train_pred = regressor.predict(X_train)
    Y_val_pred = regressor.predict(X_val)
    Y_test_pred = regressor.predict(X_test)

    # Calculate and print metrics for training set
    rmse_train = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
    mae_train = mean_absolute_error(Y_train, Y_train_pred)
    r2_train = r2_score(Y_train, Y_train_pred)
    print('Training set evaluation metrics:')
    print('RMSE: {}'.format(rmse_train))
    print("MAE: ", mae_train)
    print("R2: ", r2_train)

    # Calculate and print metrics for validation set
    rmse_val = np.sqrt(mean_squared_error(Y_val, Y_val_pred))
    mae_val = mean_absolute_error(Y_val, Y_val_pred)
    r2_val = r2_score(Y_val, Y_val_pred)
    print('\nValidation set evaluation metrics:')
    print('RMSE: {}'.format(rmse_val))
    print("MAE: ", mae_val)
    print("R2: ", r2_val)

    # Calculate and print metrics for test set
    rmse_test = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
    mae_test = mean_absolute_error(Y_test, Y_test_pred)
    r2_test = r2_score(Y_test, Y_test_pred)
    print('\nTest set evaluation metrics:')
    print('RMSE: {}'.format(rmse_test))
    print("MAE: ", mae_test)
    print("R2: ", r2_test)


if __name__ == "__main__":
    # Load the data
    data_train = pd.read_csv('/content/60minWindow_train_set (1).csv')
    data_test = pd.read_csv('/content/60minWindow_test_set (1).csv')
    data_val = pd.read_csv('/content/60minWindow_val_set (1).csv')

    main(data_train, data_test, data_val)
