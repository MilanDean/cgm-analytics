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

def fetch_files(all_data_file = '20230611_synthetic_T1DB_dataset.csv'):
    '''
    Fetch the synthetic dataset from data/input/synthetic_dataset/
    :return:
    '''
    path = '../data/input/synthetic_dataset/'
    all_data = pd.read_csv(os.path.join(path, all_data_file))
    all_data['timestamp'] = pd.to_datetime(all_data.Time)
    print('avg duration of recording', all_data.groupby('subject').timestamp.agg(np.ptp).mean())

    return all_data

def plot_rawCGM(sub_df, meals_and_cgm, save_flag = True):
    '''
    Function to plot raw CGM data into a timeseries figure. Used to compare interpolated CGM values against true CGM
    values provided in files. Meals are plotted in red.
    :param sub_df: Original CGM dataframe
    :param meals_and_cgm: Interpolated CGM dataframe
    :param save_flag: flag to determine if you want to save individual figures.
    :return: None
    '''
    plt.figure(figsize=(20, 5))
    plt.scatter(sub_df.timestamp, sub_df.CGM, label='raw_glucose')
    plt.scatter(meals_and_cgm[meals_and_cgm.CHO > 0].timestamp,
                meals_and_cgm[meals_and_cgm.CHO > 0].Glucose, color='red', label='meal')
    plt.plot(meals_and_cgm.timestamp, meals_and_cgm.Glucose, color='k', label='joined_glucose')
    plt.xlabel('time')
    plt.ylabel('glucose')
    plt.legend()
    if save_flag:
        os.makedirs('../data/output/synthetic_dataset_timeseries/', exist_ok=True)
        plt.savefig('../data/output/synthetic_dataset_timeseries/{}_cgm_timeseries.png'.format(subject))
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    # Load CGM data
    all_data = fetch_files(all_data_file = '20230611_synthetic_T1DB_dataset.csv')
    interpolated_data = []
    for subject in all_data.subject.unique().tolist():
        save_individual_files = True
        sub_df = all_data[all_data.subject == subject]
        # Interpolate CGM - create daterange of 1min intervals
        interp_dates = pd.DataFrame(pd.date_range(sub_df.timestamp.iloc[0], sub_df.timestamp.iloc[-1], freq='1min'),
                                    columns=['timestamp'])

        interp_cgm = pd.merge(sub_df, interp_dates, on=['timestamp'],
                              how='outer').sort_values('timestamp').reset_index(drop=True)
        preped_df = interp_cgm_data(interp_cgm, cgm_col='CGM', time_col='timestamp')
        preped_df['subject'] = subject

        preped_df['timestamp'] = preped_df['timestamp'] = [x.round(freq='T') for x in tqdm(preped_df.Time)]
        meals_and_cgm = pd.merge_asof(preped_df, sub_df[['BG', 'CGM', 'CHO', 'insulin', 'LBGI', 'HBGI', 'Risk',
       'timestamp']], on=['timestamp'],
                                      direction='nearest', tolerance=pd.Timedelta('29 sec'))  # 30 sec orig
        meals_and_cgm['CHO'] = meals_and_cgm['CHO'].fillna(0)

        plot_rawCGM(sub_df, meals_and_cgm)

        # save data
        os.makedirs('../data/output/synthetic_dataset_raw_wMeals/', exist_ok=True)
        if save_individual_files:
            meals_and_cgm = meals_and_cgm.drop_duplicates('timestamp')
            meals_and_cgm.to_csv('../data/output/synthetic_dataset_raw_wMeals/{}_cgm_timeseries.csv'.format(subject), index = False)

        interpolated_data.append(meals_and_cgm)

    interpolated_data_df = pd.concat(interpolated_data)
    interpolated_data_df.to_csv('../data/output/synthetic_dataset_raw_wMeals/synthetic_cgm_timeseries_allData.csv',
                                index = False)


