import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import seaborn as sns
import os
from modeling.modeling_util import *

def split_train_test(df, test_size = 0.2):
    # Train test val split on subject ID level
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=1)
    split = splitter.split(df, groups=df['subject'])
    train_inds, test_inds = next(split)
    train_df = df.iloc[train_inds]
    test_df = df.iloc[test_inds]
    print(train_df.shape, test_df.shape)
    print('| test_subjects:', test_df.subject.nunique(), '| train_subjects:', train_df.subject.nunique())
    print(sorted(test_df.subject.unique()), sorted(train_df.subject.unique()))
    print('Subjects from train set in test set: ',
          [x for x in train_df.subject.unique() if x in test_df.subject.unique()])

    return train_df, test_df

def drop_high_correlated_features(clean_df):
    # Drop features with high correlations
    plt.figure(figsize=(30, 30))
    corr_mat = clean_df.corr()
    # remove highly correlated features
    upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.90)]
    sns.heatmap(clean_df.drop(to_drop, axis=1).corr(), annot = True)
    plt.tight_layout()
    os.makedirs('../data/output/corrplots/', exist_ok = True)
    plt.savefig('../data/output/corrplots/baseline_logreg_corrplot_features.png')
    plt.close()

    return to_drop

def set_up_train_test_data(train_df, test_df):
    train_X = train_df.iloc[:, 5:]
    train_Y = train_df.meal

    test_df = test_df.rename(columns={'label(meal)':'meal'})
    test_X = test_df.iloc[:, 5:]
    test_Y = test_df.meal

    print('checking balanced labels')
    print('\t train', np.unique(train_Y, return_counts=True))
    print('\t test', np.unique(test_Y, return_counts=True))

    # Scale features
    standScale = StandardScaler()
    train_X = standScale.fit_transform(train_X)
    test_X = standScale.transform(test_X)

    # Create new DF
    train_X = pd.DataFrame(train_X, columns=train_df.iloc[:, 5:].columns)
    test_X = pd.DataFrame(test_X, columns=train_df.iloc[:, 5:].columns)
    print('ratio of training labels: ', np.unique(train_Y, return_counts=True))

    return train_X, train_Y, test_X, test_Y, standScale


if __name__ == '__main__':
    balance = True
    df = pd.read_csv('../data/output/features/synthetic_dataset_features_30minWindow.csv')
    df.head()

    # Clean Data
    df = df[~df.subject.str.contains('9')]

    # Data Cleaning
    print('shape of data: ', df.shape)
    print()
    print('NA Values')
    print(pd.DataFrame(df.isnull().sum()))
    df = df.dropna(axis=0)  # Drop NA values
    print()

    print('number of unique subjects: ', df.subject.nunique())
    df.head()
    clean_df = df.iloc[:, 5:]  # only features, not the indexing stuff
    print('shape of new dataframe: ', clean_df.shape)
    clean_df.dtypes

    # Train-Test Split
    # Balance
    df = df.rename(columns={'label(meal)': 'meal'})
    if balance == True:
        df = balance_onSubject(df)
    else:
        pass
    train_df, test_df = split_train_test(df, test_size=0.3)
    val_df, test_df = split_train_test(test_df, test_size=0.5)

    # Drop Highly Correlated Features
    to_drop = drop_high_correlated_features(clean_df)
    train_df = train_df.drop(to_drop, axis=1)
    test_df = test_df.drop(to_drop, axis=1)
    val_df = val_df.drop(to_drop, axis=1)

    # format train and test datasets correctly
    train_X, train_Y, test_X, test_Y, standScale = set_up_train_test_data(train_df, test_df)
    val_df = val_df.rename(columns={'label(meal)':'meal'})
    val_X = val_df.iloc[:, 5:]
    val_Y = val_df.meal
    val_X = pd.DataFrame(standScale.transform(val_X), columns = train_X.columns)

    save_path = '../data/output/features/'
    os.makedirs(save_path, exist_ok=True)

    scaled_train = train_X.copy()
    scaled_train['subject'] = train_df.subject.values
    scaled_train['meal'] = train_Y.values
    scaled_train.to_csv(os.path.join(save_path, '30minWindow_train_set.csv'), index = False)

    scaled_test = test_X.copy()
    scaled_test['subject'] = test_df.subject.values
    scaled_test['meal'] = test_Y.values
    scaled_test.to_csv(os.path.join(save_path, '30minWindow_test_set.csv'), index=False)

    scaled_val = val_X.copy()
    scaled_val['subject'] = val_df.subject.values
    scaled_val['meal'] = val_Y.values
    scaled_val.to_csv(os.path.join(save_path, '30minWindow_val_set.csv'), index=False)