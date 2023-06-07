import pandas as pd
import numpy as np
import os
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

def time_feature(data):
    majority_hour = data.timestamp.dt.hour.value_counts().reset_index().sort_values('timestamp',
                                                                                    ascending=False)['index'].iloc[0]
    if majority_hour < 4:
        time_feat = 0
    elif (majority_hour >= 4) & (majority_hour < 8):
        time_feat = 1
    elif (majority_hour >= 8) & (majority_hour < 12):
        time_feat = 2
    elif (majority_hour >= 12) & (majority_hour < 16):
        time_feat = 3
    elif (majority_hour >= 16) & (majority_hour < 20):
        time_feat = 4
    elif (majority_hour >= 20) & (majority_hour <= 24):
        time_feat = 5
    else:
        raise ValueError

    return time_feat

def meals_before_after(df, columns = ['mean',
       'median', 'minimum', 'maximum', 'first_quartile', 'third_quartile',
       'interday_std', 'interdaycv', 'TOR', 'TIR', 'POR', 'LBGI', #'J_index', 'eA1c',
       'HBGI', 'ADRR', 'GMI',  'entropy', 'time_period', 'ROC_min', 'ROC_max',
       'ROC_mean', 'ROC_median', 'low_range', 'target_range', 'high_range', #, 'std'
       'dynamic_risk_min', 'dynamic_risk_max', 'dynamic_risk_mean', 'dynamic_risk_median'
                                      ]):

    '''
    # excluding 'n_samples', 'meal' from columns - do not need and label is meal

    Get features related to previous and next meal. This is based on shifting features ±1 row
    :param df: dataframe of features derived from CGM (on subject level!)
    :param columns: default columns for features to use for shifts
    :return: combined dataframe of next meal and previous meal features
    '''
    meals_before = df[columns].shift(1) # shift results down 1 to get previous meal data
    meals_before.columns = ['{}_before'.format(x) for x in columns]
    meals_after = df[columns].shift(-1) # shift results down 1 to get next meal data
    meals_after.columns = ['{}_after'.format(x) for x in columns]

    combined = pd.concat([meals_before, meals_after], axis = 1)

    return combined

def derive_cgm_features(data, plot_flag=False):
    '''
    cgmquantify documentation: https://github.com/brinnaebent/cgmquantify/wiki/User-Guide
    or use https://cerebralcortex-kernel.readthedocs.io/en/latest/_modules/cerebralcortex/algorithms/glucose/glucose_variability_metrics.html
    Need to have a dataframe with 'glucose' column and time column
    :param data:
    :param plot_flag:
    :return:
    '''

    # summary_features = pd.DataFrame([cgmq.summary(data)])
    # summary_features.columns = ['mean', 'median', 'minimum', 'maximum', 'first_quartile', 'third_quartile']

    # derivative features
    firstOrdDeriv = pd.DataFrame([data['Glucose'].diff().min(), data['Glucose'].diff().max(),
                                  data['Glucose'].diff().mean(), data['Glucose'].diff().median()]).T
    firstOrdDeriv.columns = ['ROC_min', 'ROC_max', 'ROC_mean', 'ROC_median']

    # time in ranges (in minutes)
    very_high_range = data[data.Glucose > 250].shape[0] # (>250 mg/dl)
    high_range = data[(data.Glucose <= 250) & (data.Glucose >= 181)].shape[0] # (181-250 mg/dl)
    target_range = data[(data.Glucose <= 180) & (data.Glucose >= 70)].shape[0] # (70-180 mg/dl)
    low_range = data[(data.Glucose <= 69) & (data.Glucose >= 55)].shape[0] # (55-69 mg/dl)
    very_low_range = data[(data.Glucose <= 54)].shape[0] # (<54 mg/dl)

    cgm_ranges = pd.DataFrame([very_low_range, low_range, target_range, high_range, very_high_range]).T
    cgm_ranges.columns = ['very_low_range','low_range', 'target_range', 'high_range', 'very_high_range']

    # dynamic risk
    dynamic_risk_vec = dynamic_risk(glucose=data['Glucose'])
    dynamicRisk_df = pd.DataFrame([dynamic_risk_vec.min(), dynamic_risk_vec.max(),
                                  dynamic_risk_vec.mean(), dynamic_risk_vec.median()]).T
    dynamicRisk_df.columns = ['dynamic_risk_min', 'dynamic_risk_max', 'dynamic_risk_mean', 'dynamic_risk_median']

    obj = {'mean': data.Glucose.mean(),
           'median': data.Glucose.median(),
           'min': data.Glucose.min(),
           'max': data.Glucose.max(),
           'std': data.Glucose.std(),
           'first_quartile': data.Glucose.quantile(0.25),
           'third_quartile': data.Glucose.quantile(0.75),
        #'interday_std': cgmq.interdaysd(data),
           #'interdaycv': cgmq.interdaycv(data),
           # 'intradaysd': cgmq.intradaysd(data) # Does Not Work
           # 'intradaycv': cgmq.intradaycv(data) # Does Not Work
           #'TOR': cgmq.TOR(data, sd=1, sr=1)/ len(data), # change to 1? # sr is the sampling rate inverse in minutes of the CGM
           #'TIR': cgmq.TIR(data, sd=1, sr=1)/ len(data), # changing to a %
           #'POR': cgmq.POR(data, sd=1, sr=1), # change to 1?
           # 'MGE': cgmq.MGE(data, sd=1), # Does Not Work
           # 'MGN': cgmq.MGN(data, sd=1), # Does Not Work
           # 'MAGE': cgmq.MAGE(data, std=1) # Does Not Work, - same as CV
           # 'J_index': cgmq.J_index(data), # removed 12/12/22
           #'LBGI': cgmq.LBGI(data),
           #'HBGI': cgmq.HBGI(data),
           #'ADRR': cgmq.ADRR(data),
           # 'MODD': cgmq.MODD(data), # Does Not Work
           # 'CONGA24': cgmq.CONGA24(data) # Does Not Work
           #'GMI': cgmq.GMI(data),
           # 'eA1c': cgmq.eA1c(data), # Removed 12/12/22
           # 'std': np.std(data.Glucose), # repeated above
           'entropy': stats.entropy(pk=data['Glucose'], base=2),
           #'time_period': time_feature(data)
           }
    obj_df = pd.DataFrame([obj])
    #combined_df = pd.concat([summary_features, obj_df, firstOrdDeriv, cgm_ranges, dynamicRisk_df], axis = 1)
    combined_df = pd.concat([obj_df, firstOrdDeriv, cgm_ranges, dynamicRisk_df], axis=1)

    # # plotting
    # if plot_flag:
    #     cgmq.plotglucosebounds(data, upperbound=180, lowerbound=70, size=15)
    #     cgmq.plotglucosesd(data, sd=1, size=15)
    #     cgmq.plotglucosesmooth(data, size=15)

    return combined_df

def dynamic_risk(glucose, d=3.5, offset=0.75, aalpha=5):
    '''
    # adapted from Nunzio Function

    % Function for the calculation of static risk and dynamic risk.
    % INPUT:
    % glucose: vector of glucose samples [mg/dl]
    % glucose_derivative: derivative of glucose [mg/dl/min]
    % d: parameter of dynamic risk function related to maximum amplification of static risk (suggested value d=3.5)
    % offset: parameter of dynamic risk function related to maximum dumping (suggested value offset=0.75)
    % aalpha: parameter of dynamic risk function related to derivative dependent amplification (suggested value aalpha=5, original value aaplha=3)
    % OUTPUT:
    % SR: static risk
    % DR: dynamic risk
    '''

    # estimate static risk
    # SR = np.zeros(len(glucose)) # Dont need because specify below

    glucose_derivative = glucose.diff()

    # Parameters of Kovatchevs risk function
    alpha = 1.084
    beta = 5.381
    gamma = 1.509

    rl = np.zeros(len(glucose))
    rh = np.zeros(len(glucose))

    for i in range(len(glucose)):
    # for i=1:length(glucose)
        f = gamma*(((np.log(glucose.iloc[i]))**alpha)-beta)
        if f<0:
            rl[i] = 10*(f**2)
        elif f>0:
            rh[i] = 10*(f**2)
        else:
            pass
    # end

    SR = rh-rl

    # %% Calculation of the modulation factor

    modulation_factor = np.ones(len(glucose))

    # %Parameters of dynamic risk function
    ddelta=(d-offset)/2
    bbeta=ddelta+offset
    ggamma=np.arctanh((1-bbeta)/ddelta) # TODO: this is atanh = arctanh right?

    dr_over_dg = 10*gamma**2*2*alpha*(np.log(glucose)**(2*alpha-1)-beta*np.log(glucose)**(alpha-1))/glucose
    modulation_factor = ddelta*np.tanh(aalpha*dr_over_dg*glucose_derivative+ggamma)+bbeta

    # %% Calculation of dynamic risk (DR)

    DR=SR*modulation_factor

    return DR

if __name__ == '__main__':
    all_subjects_featureset= []
    path = '../data/output/hall_raw_wMeals/'
    for file in os.listdir(path):
        if not file.endswith('.csv'):
            continue
        print(file)
        df = pd.read_csv(os.path.join(path, file))
        # if df.userID.unique()[0] == '2133-032':
        #     print('s')
        # else:
        #     continue
        df['timestamp'] = pd.to_datetime(df.timestamp)
        df['day'] = [x.date().strftime(format='%Y-%m-%d') for x in df.timestamp]
        featureset = []

        if df.dropna().empty:
            continue

        for day in df.day.unique().tolist():
            day_df = df[df.day == day]
            meals = day_df.dropna()
            if meals.empty:
                continue
            start_time_day = meals.timestamp.iloc[0]
            end_time =  meals.timestamp.iloc[-1]
            negative_class = day_df[(day_df.timestamp < start_time_day) &
                                    (day_df.timestamp > start_time_day - pd.to_timedelta(60*4, 'min'))]
            negative_class['meal_flag'] = 0
            positive_class = day_df[(day_df.timestamp >= start_time_day) &
                                    (day_df.timestamp <= end_time)]
            positive_class['meal_flag'] = 1
            training_data = pd.concat([negative_class, positive_class])
            sns.lineplot(data=training_data, x='timestamp', y='Glucose', hue='meal_flag')
            plt.close()

            # create sliding window for feature derivation
            start_time = training_data.timestamp.iloc[0]
            while start_time <= training_data.timestamp.iloc[-1]:
                try:
                    # expecting 30 samples per grouping
                    sub_df = training_data[(training_data.timestamp >= start_time) & (training_data.timestamp < (start_time +
                                                                                            pd.to_timedelta(30, 'min')))]
                    sub_df = sub_df.sort_values('timestamp')
                    features = derive_cgm_features(sub_df)
                    start_block = sub_df.timestamp.iloc[0]
                    end_block = sub_df.timestamp.iloc[-1]
                    meta_info = pd.DataFrame([{'subject': training_data.userID.unique()[0],
                                               'start_block': start_block,
                                               'end_block': end_block,
                                               'n_samples': len(sub_df),
                                               'label(meal)': [1 if sum(sub_df.meal_flag) >
                                                                    sub_df.meal_flag.sum()*0.5 else 0][0]}])
                    combined_features = pd.concat([meta_info, features], axis=1)
                    # if len(combined_features) == 0:
                    #     print('s')
                    featureset.append(combined_features)
                    start_time = start_time + pd.to_timedelta(30, 'min')
                except:
                    print('stop')
        all_features = pd.concat(featureset)
        all_subjects_featureset.append(all_features)
    full_featureset = pd.concat(all_subjects_featureset)
    os.makedirs('../data/output/features/', exist_ok=True)
    full_featureset.to_csv('../data/output/features/features_20230605.csv', index = False)