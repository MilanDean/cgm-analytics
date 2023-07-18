import matplotlib.pyplot as plt

from modeling.modeling_util import *
# from preprocessing.train_test_split import *
from features import *
# from preprocessing.process_syntheticDataset import *
import pickle
import timeit

def raw_to_interp(sub_df):
    sub_df['timestamp'] = pd.to_datetime(sub_df.Time)
    interp_dates = pd.DataFrame(pd.date_range(sub_df.timestamp.iloc[0], sub_df.timestamp.iloc[-1], freq='1min'),
                                columns=['timestamp'])

    interp_cgm = pd.merge(sub_df, interp_dates, on=['timestamp'],
                          how='outer').sort_values('timestamp').reset_index(drop=True)
    preped_df = interp_cgm_data(interp_cgm, cgm_col='CGM', time_col='timestamp')
    preped_df['subject'] = 'subject'

    preped_df['timestamp'] = [x.round(freq='T') for x in tqdm(preped_df.Time)]
    meals_and_cgm = pd.merge_asof(preped_df, sub_df[['BG', 'CGM', 'CHO', 'insulin', 'LBGI', 'HBGI', 'Risk',
                                                     'timestamp']], on=['timestamp'],
                                  direction='nearest', tolerance=pd.Timedelta('29 sec'))  # 30 sec orig
    meals_and_cgm['CHO'] = meals_and_cgm['CHO'].fillna(0)

    return meals_and_cgm

def interp_cgm_data(df, cgm_col='cgm_val', time_col='timestamps'):
    '''
    pandas interpolate: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html

    :param df:
    :param cgm_col:
    :return:
    '''
    data = pd.DataFrame()
    data['Time'] = df[time_col] # need a time column
    data['Time'] = pd.to_datetime(data['Time'])#, format='%Y-%m-%dT%H:%M:%S')
    data['Glucose'] = df[cgm_col].interpolate(method='pchip')  # interpolate missing values
    data['Day'] = data['Time'].dt.date  # need a day column
    #data = data.reset_index()

    return data

def derive_features(df, subject, window_size = 60, shift = 60):
    df['day'] = [x.date().strftime(format='%Y-%m-%d') for x in df.timestamp]
    featureset = []

    # Remove completely empty data
    if df[['timestamp', 'Glucose', 'CHO']].dropna().empty:
        print(' \t No Data Present')
        raise ValueError

    # create sliding window for feature derivation
    start_time = df.timestamp.iloc[0]
    while start_time <= df.timestamp.iloc[-1]:
        try:
            # expecting 30 samples per grouping
            sub_df = df[(df.timestamp >= start_time) & (df.timestamp < (start_time + pd.to_timedelta(window_size, 'min')))]
            sub_df = sub_df.sort_values('timestamp')
            features = derive_cgm_features(sub_df)
            start_block = sub_df.timestamp.iloc[0]
            end_block = sub_df.timestamp.iloc[-1]
            meta_info = pd.DataFrame([{'subject': subject,
                                       'start_block': start_block,
                                       'end_block': end_block,
                                       'n_samples': len(sub_df),
                                       'label(meal)': np.where(sub_df.CHO.sum() > 0, 1, 0),
                                       'CHO_total': sub_df.CHO.max()}])
            combined_features = pd.concat([meta_info, features], axis=1)
            # if len(combined_features) == 0:
            #     print('s')
            featureset.append(combined_features)
            start_time = start_time + pd.to_timedelta(shift, 'min')
        except:
            print('stop')
    all_features = pd.concat(featureset)
    prePost_meal_features = meals_before_after(all_features)
    join_wSub_df = pd.concat([all_features, prePost_meal_features], axis=1)
    full_featureset = join_wSub_df[join_wSub_df.n_samples == window_size]

    return full_featureset

FEATURE_NAMES = ['mean', 'std', 'interdaycv', 'TOR', 'TIR', 'PIR', 'LBGI', 'time_period',
       'ROC_min', 'ROC_max', 'ROC_mean', 'very_low_range', 'low_range',
       'target_range', 'high_range', 'very_high_range', 'dynamic_risk_min',
       'dynamic_risk_max', 'dynamic_risk_mean', 'mean_before', 'std_before',
       'interdaycv_before', 'TOR_before', 'TIR_before', 'PIR_before',
       'LBGI_before', 'time_period_before', 'ROC_min_before', 'ROC_max_before',
       'ROC_mean_before', 'very_low_range_before', 'low_range_before',
       'target_range_before', 'high_range_before', 'very_high_range_before',
       'dynamic_risk_min_before', 'dynamic_risk_max_before',
       'dynamic_risk_mean_before', 'mean_after', 'std_after',
       'interdaycv_after', 'TOR_after', 'TIR_after', 'PIR_after', 'LBGI_after',
       'time_period_after', 'ROC_min_after', 'ROC_max_after', 'ROC_mean_after',
       'very_low_range_after', 'low_range_after', 'target_range_after',
       'high_range_after', 'very_high_range_after', 'dynamic_risk_min_after',
       'dynamic_risk_max_after', 'dynamic_risk_mean_after', 'age', 'subject',
       'meal', 'start_block', 'end_block', 'CHO_total']

def preprocess_data(file):
    # preprocess
    interp_data = raw_to_interp(file)
    feature_set = derive_features(interp_data, subject = 'test1')
    feature_set = feature_set.rename(columns = {'label(meal)':'meal'})
    feature_set['age'] = age
    feature_set = feature_set.dropna(axis=0)
    reduced_feature_set = feature_set[FEATURE_NAMES] # only select features without high correlation values

    return reduced_feature_set

def scale_data(reduced_feature_set):
    scalar = pickle.load(open('../data/output/models/standard_scaler_mealDetection.pickle', 'rb'))
    scaled_features = scalar.transform(reduced_feature_set.iloc[:, :-5])
    scaled_features = pd.DataFrame(scaled_features)
    scaled_features.columns = reduced_feature_set.iloc[:, :-5].columns
    scaled_features['subject'] = reduced_feature_set.subject.values
    scaled_features['meal'] = reduced_feature_set.meal.values
    scaled_features['start_block'] = reduced_feature_set.start_block.values
    scaled_features['end_block'] = reduced_feature_set.end_block.values
    scaled_features['CHO_total'] = reduced_feature_set.CHO_total.values

    return scaled_features

def plot_daily_predictions(meal_data):
    # Display Results
    meal_data['start_block'] = pd.to_datetime(meal_data.start_block)
    meal_data['day'] = meal_data.start_block.dt.date
    meals_per_day = meal_data.groupby('day')[['predictions', 'carb_preds']].sum().reset_index()
    meals_per_day.columns = ['Day', 'Meals', 'Total Carbs']
    plt.subplot(211)
    plt.bar(meals_per_day.Day, meals_per_day.Meals)
    plt.xticks(rotation = 45)
    plt.ylabel('Meals per Day')
    plt.subplot(212)
    plt.bar(meals_per_day.Day, meals_per_day['Total Carbs'])
    plt.xticks(rotation = 45)
    plt.ylabel('Total CHO (g) per Day')
    plt.axhline(y=225, color = 'r', linestyle='--')
    plt.axhline(y=325, color='r', linestyle='--',label='Healthy Carb Amount')
    plt.legend()
    plt.tight_layout()

    plt.savefig('./output_of_NutriNet_testing.png')
    
def generate_meal_diary_table(meal_data):
    meal_data['Date'] = [pd.to_datetime(x).date().strftime(format='%m/%d/%Y') for x in meal_data.start_block]
    meal_data['Meal Time'] = [pd.to_datetime(x).time().strftime(format='%H:%M') for x in meal_data.start_block]
    meal_data['Carbs (g)'] = round(meal_data.carb_preds)
    conditions = [meal_data.Carbs < 28,
                  (meal_data.Carbs >= 28) & (meal_data.Carbs < 75),
                  (meal_data.Carbs >= 75)]
    selections = ['small meal', 'light/medium meal', 'large meal']
    meal_data['Meal Size'] = np.select(conditions, selections)
    display_df = meal_data[['Date', 'Meal Time', 'Carbs (g)', 'Meal Size']]
    display_df.to_csv('./meal_prediction_table.csv', index = False)

if __name__ == '__main__':
    file = pd.read_csv('../data/input/synthetic_dataset/results/adult#001.csv')

    meal_detect_model = pickle.load(open('../data/output/models/lgbm_mealDetection_model.pickle', 'rb'))
    carb_estimate_model = pickle.load(open('../data/output/models/svr_model_carbEstimate.pickle', 'rb'))
    rfe_results = pd.read_csv('../data/output/training/training_20230702/tuned_to_precision/60minWindow/lgbm_features_20230709.csv')
    age = input('enter age of participant: ')
    # Timeit - start
    starttime = timeit.default_timer()

    # preprocess
    reduced_feature_set = preprocess_data(file)

    # Scale Data
    scaled_features = scale_data(reduced_feature_set)

    # Select features from RFE
    selected_features = rfe_results.feature.to_list()
    optimal_threshold = 0.62062

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

    # End time
    print("Duration of Code Execution :", timeit.default_timer() - starttime)

    # Eval of Predictions
    print('confusionMatrix Meal Detection')
    cm = confusion_matrix(scaled_features.meal.astype('int'), scaled_features['predictions'].astype('int'))
    ConfusionMatrixDisplay(cm).plot()

    # Display Results
    plot_daily_predictions(meal_data)

    # Meal Tables
    generate_meal_diary_table(meal_data)
    