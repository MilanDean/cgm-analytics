from helper_functions import *

import argparse
import timeit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='NutriNet',
        description='Program for performing NutriNet processing and Modeling',
        epilog='TBD....')
    parser.add_argument('--file', type=str, help='path to CGM file')
    #file = pd.read_csv('../data/input/synthetic_dataset/results/adult#001.csv')
    scalar = pickle.load(open('../data/output/models/standard_scaler_mealDetection.pickle', 'rb'))
    meal_detect_model = pickle.load(open('../data/output/models/lgbm_mealDetection_model.pickle', 'rb'))
    carb_estimate_model = pickle.load(open('../data/output/models/svr_model_carbEstimate.pickle', 'rb'))
    rfe_results = pd.read_csv('../data/output/training/training_20230702/tuned_to_precision/60minWindow/lgbm_features_20230709.csv')
    parser.add_argument('--age', type=str, help='Age of CGM Data Participant')
    #age = input('enter age of participant: ')

    ## Parse Args
    parser.parse_args(['--file', '--age'])

    # Timeit - start
    starttime = timeit.default_timer()

    # preprocess
    reduced_feature_set = preprocess_data(file)

    # Scale Data
    scaled_features = scale_data(reduced_feature_set)

    # Select features from RFE
    selected_features = rfe_results.feature.to_list()
    optimal_threshold = 0.2793919

    # Set-up Prediction dataset
    X = scaled_features.iloc[:,:-5]
    X = X[selected_features]

    # Make predictions for meals
    preds = meal_detect_model.predict(X)
    scaled_features['predictions'] = preds

    # Now do the carb estimation
    meal_data = scaled_features[scaled_features.predictions == 1]
    carbEst_results = pd.read_csv('../data/output/training/svr_features_20230710.csv')
    carbEst_features = carbEst_results.feature.to_list()
    X = meal_data[carbEst_features]
    preds = carb_estimate_model.predict(X)
    meal_data['carb_preds'] = preds

    # Save Outputs
    plot_daily_predictions(meal_data)

    # Meal Tables
    generate_meal_diary_table(meal_data)
