

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor, ExtraTreesRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
import pickle

def load_data(train_path, test_path, val_path):
    data_train = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)
    data_val = pd.read_csv(val_path)
    return data_train, data_test, data_val

def preprocess_data(data_train, data_test, data_val):
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
    X_test = data_test.drop([ 'subject','meal', 'start_block', 'end_block','CHO_total'], axis=1)
    Y_test = data_test['CHO_total']
    X_val = data_val.drop(['subject','meal', 'start_block', 'end_block','CHO_total'], axis=1)
    Y_val = data_val['CHO_total']

    return X_train, Y_train, X_test, Y_test, X_val, Y_val

def train_and_evaluate(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    # Define the base models
    base_models = [
        ('extra_trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
        ('catboost', CatBoostRegressor(logging_level='Silent', random_state=42)),
        ('random_forest', RandomForestRegressor(n_estimators=100, random_state=42))
    ]

    # Define the stacking ensemble
    stacked_ensemble = StackingRegressor(estimators=base_models, final_estimator=Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None, positive=False, random_state=None, solver='auto', tol=0.001))

    # Train the ensemble
    stacked_ensemble.fit(X_train, Y_train)

    # Make predictions
    y_pred = stacked_ensemble.predict(X_val)

    # Evaluate the model
    mse = mean_squared_error(Y_val, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_val, y_pred)
    r2 = r2_score(Y_val, y_pred)

    print(f'Validation set metrics:')
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R2 Score: {r2}')

    # Save the model
    pkl_filename = "pickle_model2_ensemble_regr.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(stacked_ensemble, file)

    # Use the model to make predictions on the test data
    y_test_pred = stacked_ensemble.predict(X_test)

    # Calculate the metrics
    mse_test = mean_squared_error(Y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    mae_test = mean_absolute_error(Y_test, y_test_pred)
    r2_test = r2_score(Y_test, y_test_pred)

    print(f'\nTest set metrics:')
    print(f'Mean Squared Error: {mse_test}')
    print(f'Root Mean Squared Error: {rmse_test}')
    print(f'Mean Absolute Error: {mae_test}')
    print(f'R2 Score: {r2_test}')

if __name__ == "__main__":
    train_path = '/content/60minWindow_train_set (1).csv'
    test_path = '/content/60minWindow_test_set (1).csv'
    val_path = '/content/60minWindow_val_set (1).csv'
    data_train, data_test, data_val = load_data(train_path, test_path, val_path)
    X_train, Y_train, X_test, Y_test, X_val, Y_val = preprocess_data(data_train, data_test, data_val)
    train_and_evaluate(X_train, Y_train, X_val, Y_val, X_test, Y_test)
