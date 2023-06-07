import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    # Load CGM data
    path = '../data/input/Hall_2018_Data/'
    all_data_file = 'pbio.2005143.s010'
    all_data = pd.read_table(os.path.join(path, all_data_file))
    all_data = all_data.rename(columns={'DisplayTime': 'time',
                             'subjectId': 'userID'})
    all_data['timestamp'] = pd.to_datetime(all_data.time)
    print('avg duration of recording', all_data.groupby('userID').timestamp.agg(np.ptp).mean())

    # meal data
    meal_data_file = 'pbio.2005143.s015.tsv'
    meal_data = pd.read_table(os.path.join(path, meal_data_file))
    conditions = [meal_data.Meal.str.contains('PB'),
                  meal_data.Meal.str.contains('CF'),
                  meal_data.Meal.str.contains('Bar')
                  ]
    selections1 = [430, 370, 280]
    selections2 = [20, 18, 2.5]
    selections3 = [51, 48, 54]
    selections4 = [12, 19, 35.2]
    selections5 = [12, 6, 3.3]
    selections6 = [18, 9, 11]
    meal_data['calories'] = np.select(conditions, selections1)
    meal_data['fat'] = np.select(conditions, selections2)
    meal_data['carbs'] = np.select(conditions, selections3)
    meal_data['sugar'] = np.select(conditions, selections4)
    meal_data['fiber'] = np.select(conditions, selections5)
    meal_data['protein'] = np.select(conditions, selections6)
    meal_data['meal_flag'] = 1
    meal_data['timestamp'] = pd.to_datetime(meal_data.time)
    meal_data['timestamp'] = [x.round(freq='T') for x in tqdm(meal_data.timestamp)]
    meal_data['GlucoseValue'] = meal_data['GlucoseValue'].replace('Low',
                                                            meal_data.GlucoseValue.min())
    meal_data['GlucoseValue'] = meal_data['GlucoseValue'].astype('float')

    # # Join with cgm data
    # meals_and_cgm = meal_data_interp.merge(meal_data,
    #                 on = ['time', 'userID'], how = 'left')
    #

    outputs = []
    for subject in all_data.userID.unique().tolist():
        sub_df = all_data[all_data.userID ==  '2133-032']#subject]
        sub_df = sub_df.sort_values('timestamp')
        # get meal data
        sub_meals = meal_data[meal_data.userID == subject]
        if sub_meals.empty:
            continue

        sub_df['GlucoseValue'] = sub_df['GlucoseValue'].replace('Low',
                                            sub_df.GlucoseValue.min())
        sub_df['GlucoseValue'] = sub_df['GlucoseValue'].astype('float')

        # Interpolate CGM - create daterange of 1min intervals
        interp_dates = pd.DataFrame(pd.date_range(sub_df.timestamp.iloc[0], sub_df.timestamp.iloc[-1], freq='1min'),
                                    columns=['timestamp'])
        #interp_dates['timestamps'] = [convert_timestamps(x) for x in interp_dates.timestamps]

        interp_cgm = pd.merge(sub_df, interp_dates, on=['timestamp'],
                              how='outer').sort_values('timestamp').reset_index(drop=True)
        preped_df = interp_cgm_data(interp_cgm, cgm_col='GlucoseValue', time_col='timestamp')
        preped_df['userID'] = subject
        #preped_df = preped_df.rename(columns = {'Time': 'time'})

        preped_df['timestamp'] = preped_df['timestamp'] = [x.round(freq='T') for x in tqdm(preped_df.Time)]
        meals_and_cgm = pd.merge_asof(preped_df, sub_meals[['Meal', 'timestamp', 'calories', 'fat', 'carbs',
       'sugar', 'fiber', 'protein', 'meal_flag']], on = ['timestamp'],
                                    direction='nearest', tolerance=pd.Timedelta('29 sec'))  # 30 sec orig
        plt.figure(figsize = (20,5))
        plt.scatter(preped_df.timestamp, preped_df.Glucose, label = 'raw_glucose')
        plt.scatter(sub_meals.timestamp, sub_meals.GlucoseValue, color='red', label='meal')
        plt.plot(meals_and_cgm.timestamp, meals_and_cgm.Glucose, color='k', label = 'joined_glucose')
        plt.scatter(meals_and_cgm.dropna().timestamp, meals_and_cgm.dropna().Glucose, color='green', label = 'joined_meals')
        plt.xlabel('time')
        plt.ylabel('glucose')
        plt.legend()
        os.makedirs('../data/output/hall_raw_timeseries/', exist_ok=True)
        plt.savefig('../data/output/hall_raw_timeseries/{}_cgm_timeseries.png'.format(subject))
        plt.close()

        # save data
        os.makedirs('../data/output/hall_raw_wMeals/', exist_ok=True)
        meals_and_cgm = meals_and_cgm.drop_duplicates('timestamp')
        meals_and_cgm.to_csv('../data/output/hall_raw_wMeals/{}_cgm_timeseries.csv'.format(subject), index = False)
    # meal_data_interp = pd.concat(outputs)
    # meal_data_interp['time'] = [x.round(freq='T') for x in tqdm(meal_data_interp.time)]
