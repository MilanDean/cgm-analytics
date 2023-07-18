import matplotlib.pyplot as plt
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder
from modeling_util import *
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.base import clone
from probatus.feature_elimination import ShapRFECV
import pickle
import warnings
warnings.filterwarnings("ignore")

def split_train_test(df):
    # Train test val split on subject ID level
    splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=1)
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
    plt.savefig('../data/output/corrplots/baseline_lgbm_corrplot_features.png')
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

def plot_roc_curves(train_Y, train_probs, test_Y, test_probs):
    fpr_train, tpr_train, thresholds_train = metrics.roc_curve(train_Y, train_probs, pos_label=1)
    roc_auc_train = metrics.auc(fpr_train, tpr_train)
    fpr, tpr, thresholds = metrics.roc_curve(test_Y, test_probs, pos_label=1)
    roc_auc_test = metrics.auc(fpr, tpr)
    plt.plot(fpr_train, tpr_train, label = 'train ({:.2f})'.format(roc_auc_train))
    plt.plot(fpr, tpr, label='test ({:.2f})'.format(roc_auc_test), linestyle='--')
    plt.plot([0, 1], [0, 1], color="navy", lw = 2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig('../data/output/training/baseline_lgbm_ROC_{}.png'.format(
        datetime.today().strftime('%Y%m%d')))

if __name__ == '__main__':
    balance = True
    train_df = pd.read_csv('../data/output/features/60minWindow_train_set.csv')
    test_df = pd.read_csv('../data/output/features/60minWindow_test_set.csv') # I know it says test, but its val
    val_df = pd.read_csv('../data/output/features/60minWindow_val_set.csv')
    rfe_results = pd.read_csv('../data/output/training/svr_features_20230710.csv')
    selected_features = rfe_results.feature.to_list()

    # Filter for rows with meals
    train_df = train_df[train_df.meal > 0 ]
    test_df = test_df[test_df.meal > 0]
    val_df = val_df[val_df.meal > 0]

    # format train and test datasets correctly
    train_X = train_df.iloc[:,:-5]
    train_X = train_X[selected_features]
    train_Y = train_df.CHO_total * 3
    test_X = test_df.iloc[:,:-5]
    test_X = test_X[selected_features]
    test_Y = test_df.CHO_total * 3
    val_X = val_df.iloc[:,:-5]
    val_X = val_X[selected_features]
    val_Y = val_df.CHO_total * 3

    #####################
    ### Model Testing ###
    #####################

    #### Hyperparameter Tuned Models ####
    baseline_results = []
    # Create a list where train data indices are -1 and validation data indices are 0
    combined_X = pd.concat([train_X, val_X]).reset_index(drop = True)
    group_array = np.array(pd.concat([train_df, val_df]).subject)
    combined_Y = pd.concat([train_Y, val_Y]).reset_index(drop = True)

    svrF_model = SVR(degree = 1,
                    kernel='rbf',
                    gamma=0.1,
                    C=1000,
                    )

    svrF_model.fit(combined_X, combined_Y)
    yPrime = svrF_model.predict(test_X)
    baseline_results.append({'model': 'tuned_svr',
                             'mean_squared_error': mean_squared_error(test_Y, yPrime, squared=False),
                             'mean_absolute_percentage_error': mean_absolute_percentage_error(test_Y, yPrime),
                             'mean_absolute_error': mean_absolute_error(test_Y, yPrime),
                             'r2_score': r2_score(test_Y, yPrime),
                             })

    yPrime_train = svrF_model.predict(combined_X)
    baseline_results.append({'model': 'tuned_svr_train_results',
                             'mean_squared_error': mean_squared_error(combined_Y, yPrime_train, squared=False),
                             'mean_absolute_percentage_error': mean_absolute_percentage_error(combined_Y, yPrime_train),
                             'mean_absolute_error': mean_absolute_error(combined_Y, yPrime_train),
                             'r2_score': r2_score(combined_Y, yPrime_train)
                             })

    os.makedirs('../data/output/training/', exist_ok=True)

    # Save top features
    pd.DataFrame({'feature': train_X.columns,
                  'importance': 1}).sort_values('importance', ascending = False).to_csv(
        '../data/output/training/Final_svr_features_{}.csv'.format(datetime.today().strftime('%Y%m%d')))

    # Save the meal predictions
    test_df['predictions'] = yPrime
    test_df.to_csv('../data/output/training/Final_svr_predicitons_{}.csv'.format(
        datetime.today().strftime('%Y%m%d')), index = False)

    # Scatterplot of points
    plt.figure()
    r2_metric = r2_score(test_Y, yPrime)
    sns.regplot(data = test_df, x = 'CHO_total', y='predictions',
                line_kws={'color':'red'})
    plt.title('SVR True CHO vs Predictions (r2 = {:.2f})'.format(r2_metric))
    plt.savefig('../data/output/training/Final_svr_r2_plot.png')
    plt.close()
    # Save results
    results = pd.DataFrame(baseline_results).sort_values('mean_squared_error')
    results.to_csv('../data/output/training/Final_svr_results.csv', index = False)
    print(results)

    # Save models
    # os.makedirs('../data/output/models/', exist_ok=True)
    # with open('../data/output/models/svr_model_carbEstimate.pickle', 'wb') as handle:
    #     pickle.dump(svrF_model, handle, protocol=pickle.HIGHEST_PROTOCOL)
