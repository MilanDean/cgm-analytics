import pandas as pd
import numpy as np
import os
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from modeling.modeling_util import *
from derive_features import *
from pyts.image import GramianAngularField

def create_CNN_figures(sub_df, subject, label):
    plt.figure(figsize=(10,10))
    sub_df['time'] = (sub_df.timestamp-sub_df.timestamp.iloc[0]).dt.total_seconds()
    plt.plot(sub_df.time, sub_df.Glucose)
    plt.ylim([50, 350])
    plt.tight_layout
    save_path = '../data/output/features/cnn/images/'
    os.makedirs(save_path, exist_ok=True)
    filename = '{}_{}_{}.png'.format(subject, sub_df.timestamp.iloc[0].strftime(format='%Y%m%d_%H%M%S'), label)
    plt.savefig(os.path.join(save_path, filename), dpi=300)
    plt.close()

    return os.path.join(save_path, filename)

def create_gaf(sub_df, subject, label):
    """
    https://medium.com/analytics-vidhya/encoding-time-series-as-images-b043becbdbf3

    :param ts:
    :return:
    """
    ts = sub_df.Glucose
    data = dict()
    gadf = GramianAngularField(method='difference', image_size=ts.shape[0])
    data['gadf'] = gadf.fit_transform(pd.DataFrame(ts).T)[0] # ts.T)
    plt.imshow(data['gadf'], cmap='rainbow')
    width_ratios = (2, 7, 7, 0.4)
    height_ratios = (2, 7)
    # width = 10
    # height = width * sum(height_ratios) / sum(width_ratios)
    # fig = plt.figure(figsize=(width, height))
    plt.GridSpec(2, 4, width_ratios=width_ratios,
                          height_ratios=height_ratios,
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    save_path = '../data/output/features/cnn/images/'
    os.makedirs(save_path, exist_ok=True)
    filename = '{}_{}_{}.png'.format(subject, sub_df.timestamp.iloc[0].strftime(format='%Y%m%d_%H%M%S'), label)
    plt.savefig(os.path.join(save_path, filename), dpi=300)
    plt.close()

    return os.path.join(save_path, filename)

if __name__ == '__main__':
    all_subjects_featureset= []
    file = '../data/output/synthetic_dataset_raw_wMeals/20230622_synthetic_T1DB_interpolated_dataset.csv'
    all_data = pd.read_csv(file)
    all_data['timestamp'] = pd.to_datetime(all_data.timestamp) # Make datetime type

    # Read in featureset
    features = pd.read_csv('../data/output/features/synthetic_dataset_features_60minWindow_30minOverlap.csv')
    features.head()

    # empty list for storing features
    featureset = []

    for subject in tqdm(all_data.subject.unique().tolist()):
        df = all_data[all_data.subject == subject]
        df['day'] = [x.date().strftime(format='%Y-%m-%d') for x in df.timestamp]
        sub_features = features[features.subject == subject].rename(columns = {'label(meal)':'meal'})
        sub_features = balance_onSubject(sub_features)

        # Remove completely empty data
        if df[['timestamp', 'Glucose', 'CHO']].dropna().empty:
            continue

        for index, row in sub_features.iterrows():
            try:
                # expecting 30 samples per grouping
                start_time = row.start_block
                end_time = row.end_block
                sub_df = df[(df.timestamp >= start_time) & (df.timestamp <= end_time)]
                sub_df = sub_df.sort_values('timestamp')
                output_filename = create_gaf(sub_df, subject, row.meal)
                # output_filename = create_CNN_figures(sub_df, subject, np.where(sub_df.CHO.sum() > 0, 1, 0))
                # features = derive_cgm_features(sub_df)
                start_block = sub_df.timestamp.iloc[0]
                end_block = sub_df.timestamp.iloc[-1]
                meta_info = pd.DataFrame([{'subject': subject,
                                           'start_block': start_block,
                                           'end_block': end_block,
                                           'n_samples': len(sub_df),
                                           'label(meal)': row.meal,
                                           'filename': output_filename}])

                featureset.append(meta_info)
                # start_time = start_time + pd.to_timedelta(30, 'min')
            except:
                print('stop')
        # all_features = pd.concat(featureset)
        # all_subjects_featureset.append(all_features)
    full_featureset = pd.concat(featureset)
    # full_featureset = full_featureset[full_featureset.n_samples == 30]
    os.makedirs('../data/output/features/', exist_ok=True)
    full_featureset.to_csv('../data/output/features/synthetic_dataset_cnn_image_map_60minWindow_2023.csv',
                           index = False)