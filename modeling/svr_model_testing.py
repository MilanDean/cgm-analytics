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
    test_df = pd.read_csv('../data/output/features/60minWindow_val_set.csv') # I know it says test, but its val
    #val_df = pd.read_csv('../data/output/features/60minWindow_val_set.csv')

    # Filter for rows with meals
    train_df = train_df[train_df.meal > 0 ]
    test_df = test_df[test_df.meal > 0]

    # format train and test datasets correctly
    train_X = train_df.iloc[:,:-5]
    train_Y = train_df.CHO_total * 3
    test_X = test_df.iloc[:,:-5]
    test_Y = test_df.CHO_total * 3
    # val_X = val_df.iloc[:,:-1]
    # val_Y = val_df.meal

    #####################
    ### Model Testing ###
    #####################
    baseline_results = [] # Set up object for storing baseline
    # Run baseline
    mean_preds = train_Y.mean() # used to be np.ones but switched to 0s for majority class
    baseline_results.append({'model': 'predict_all_0s',
                             'mean_squared_error': mean_squared_error(test_Y, np.repeat(mean_preds, len(test_Y)), squared=False),
                             'mean_absolute_percentage_error': mean_absolute_percentage_error(test_Y,
                                                                                np.repeat(mean_preds, len(test_Y))),
                             'mean_absolute_error': mean_absolute_error(test_Y, np.repeat(mean_preds, len(test_Y))),
                             'r2_score': r2_score(test_Y, np.repeat(mean_preds, len(test_Y)))})

    #### Logistic Regression ####
    group_array = train_df.groupby(["subject"])["subject"].count().to_numpy()
    svr_model = SVR()
    svr_model.fit(train_X, train_Y)
    yPrime = svr_model.predict(test_X)
    train_preds = svr_model.predict(train_X)
    baseline_results.append({'model': 'untuned_svr',
                             'mean_squared_error': mean_squared_error(test_Y, yPrime, squared=False),
                             'mean_absolute_percentage_error': mean_absolute_percentage_error(test_Y, yPrime),
                             'mean_absolute_error': mean_absolute_error(test_Y, yPrime),
                             'r2_score': r2_score(test_Y, yPrime)})

    #### Hyperparameter Tuned Models ####
    # Create a list where train data indices are -1 and validation data indices are 0
    combined_X = pd.concat([train_X, test_X]).reset_index(drop = True)
    group_array = np.array(pd.concat([train_df, test_df]).subject)
    combined_Y = pd.concat([train_Y, test_Y]).reset_index(drop = True)
    split_index = [-1 if x in train_X.index else 0 for x in combined_X.index]
    pds = PredefinedSplit(test_fold=split_index)

    # SKLEARN RFE --- Reduce to only top features
    # selector = RFECV(estimator = SVR(), #boosting_type = 'gbdt'
    #                   scoring='neg_mean_absolute_error', step=1, cv=pds, n_jobs=-1)
    sfs = SequentialFeatureSelector(SVR(), n_features_to_select=12, scoring='r2', cv=pds, n_jobs=-1)
    sfs_out = sfs.fit(combined_X, combined_Y)
    combined_X = combined_X.iloc[:, sfs_out.support_]
    train_X = train_X.iloc[:, sfs_out.support_]
    test_X = test_X.iloc[:, sfs_out.support_]

    degree = [int(x) for x in np.linspace(1, 10, num=1)]
    kernel = ['linear', 'poly',
              'rbf', 'sigmoid']
    C = [0.1, 1., 10., 100., 1000.]
    gamma = [1, 0.1, 0.01, 0.001, 0.0001]

    # Use GridSearch
    svr_model = GridSearchCV(estimator= SVR(),
                              param_grid={'degree': degree,
                                'kernel': kernel,
                                'C': C,
                                'gamma': gamma}, n_jobs=-1, cv= pds,  #sgkf,
                              scoring='r2')

    # Fit the model
    svr_model.fit(combined_X, combined_Y, groups=group_array) #, **fit_params) # Combined
    # Used to use SVR
    svrF_model = SVR(degree = svr_model.best_estimator_.get_params()['degree'],
                    kernel=svr_model.best_estimator_.get_params()['kernel'],
                    gamma=svr_model.best_estimator_.get_params()['gamma'],
                    C=svr_model.best_estimator_.get_params()['C'],
                    )

    svr_cv_scores_mean_squared_error = cross_val_score(estimator=svrF_model, X=combined_X, y=combined_Y, groups=group_array,
                                                       cv= pds,  # sgkf,
                                                       n_jobs=-1, scoring='neg_root_mean_squared_error')
    svr_cv_scores_mean_absolute_percentage_error = cross_val_score(estimator=svrF_model, X=combined_X, y=combined_Y, groups=group_array,
                                                                   cv= pds,  #sgkf,
                                                                   n_jobs=-1, scoring='neg_mean_absolute_percentage_error')
    svr_cv_scores_mean_absolute_error = cross_val_score(estimator=svrF_model, X=combined_X, y=combined_Y, groups=group_array,
                                                        cv= pds,  #sgkf,
                                                        n_jobs=-1, scoring='neg_mean_absolute_error')
    svr_cv_scores_r2 = cross_val_score(estimator=svrF_model, X=combined_X, y=combined_Y, groups=group_array,
                                       cv= pds,  #sgkf,
                                       n_jobs=-1, scoring='r2')

    svrF_model.fit(train_X, train_Y)
    yPrime = svrF_model.predict(test_X)
    baseline_results.append({'model': 'tuned_svr',
                             'mean_squared_error': mean_squared_error(test_Y, yPrime, squared=False),
                             'mean_absolute_percentage_error': mean_absolute_percentage_error(test_Y, yPrime),
                             'mean_absolute_error': mean_absolute_error(test_Y, yPrime),
                             'r2_score': r2_score(test_Y, yPrime),
                             'cross_val_mean_squared_error': svr_cv_scores_mean_squared_error[0],
                             'cross_val_mean_absolute_error': svr_cv_scores_mean_absolute_error[0],
                             'cross_val_mean_absolute_percentage_error': svr_cv_scores_mean_absolute_percentage_error[0],
                             'cross_val_mean_r2_score': svr_cv_scores_r2[0]
                             })

    yPrime_train = svrF_model.predict(train_X)
    baseline_results.append({'model': 'tuned_svr_train_results',
                             'mean_squared_error': mean_squared_error(train_Y, yPrime_train, squared=False),
                             'mean_absolute_percentage_error': mean_absolute_percentage_error(train_Y, yPrime_train),
                             'mean_absolute_error': mean_absolute_error(train_Y, yPrime_train),
                             'r2_score': r2_score(train_Y, yPrime_train)
                             })

    os.makedirs('../data/output/training/', exist_ok=True)

    # Save top hyperparams
    pd.DataFrame([svr_model.best_params_]).to_csv('../data/output/training/svr_top_hyperparams.csv')

    # Save top features
    pd.DataFrame({'feature': train_X.columns,
                  'importance': 1}).sort_values('importance', ascending = False).to_csv(
        '../data/output/training/svr_features_{}.csv'.format(datetime.today().strftime('%Y%m%d')))

    # Save the meal predictions
    test_df['predictions'] = yPrime
    test_df.to_csv('../data/output/training/baseline_svr_predicitons_{}.csv'.format(
        datetime.today().strftime('%Y%m%d')), index = False)

    # Scatterplot of points
    plt.figure()
    r2_metric = r2_score(test_Y, yPrime)
    sns.regplot(data = test_df, x = 'CHO_total', y='predictions',
                line_kws={'color':'red'})
    plt.title('SVR True CHO vs Predictions (r2 = {:.2f})'.format(r2_metric))
    plt.savefig('../data/output/training/svr_r2_plot.png')
    plt.close()
    # Save results
    results = pd.DataFrame(baseline_results).sort_values('mean_squared_error')
    results.to_csv('../data/output/training/baseline_svr_results.csv', index = False)
    print(results)

    # Save models
    # os.makedirs('../data/output/models/', exist_ok=True)
    # with open('../data/output/models/lgbm_model.pickle', 'wb') as handle:
    #     pickle.dump(lgbmF_clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # scalerfile = '../data/output/models/lgbm_standard_scaler.pickle'
    # pickle.dump(standScale, open(scalerfile, 'wb'))