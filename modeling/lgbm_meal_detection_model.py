import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder
from modeling_util import *
from lightgbm import LGBMModel, LGBMClassifier, cv
from sklearn.feature_selection import RFECV
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
    train_df = pd.read_csv('../data/output/features/60minWindow_imbal_train_set.csv')
    test_df = pd.read_csv('../data/output/features/60minWindow_imbal_test_set.csv') # I know it says test, but its val
    val_df = pd.read_csv('../data/output/features/60minWindow_imbal_val_set.csv')
    rfe_results = pd.read_csv('../data/output/training/imbal_tuneToPrec/lgbm_features_20230716.csv') #training_20230702/tuned_to_precision/60minWindow/lgbm_features_20230709.csv')
    selected_features = rfe_results.feature.to_list()
    optimal_threshold = 0.78054 # From J-index from optimal model

    # format train and test datasets correctly
    train_X = train_df.iloc[:,:-5]
    train_X = train_X[selected_features]
    train_Y = train_df.meal
    test_X = test_df.iloc[:,:-5]
    test_X = test_X[selected_features]
    test_Y = test_df.meal
    val_X = val_df.iloc[:,:-5]
    val_X = val_X[selected_features]
    val_Y = val_df.meal

    #### Hyperparameter Tuned Models ####
    baseline_results = []
    # Create a list where train data indices are -1 and validation data indices are 0
    combined_X = pd.concat([train_X, val_X]).reset_index(drop = True)
    group_array = np.array(pd.concat([train_df, val_df]).subject)
    combined_Y = pd.concat([train_Y, val_Y]).reset_index(drop = True)

    lgbmF_clf = LGBMClassifier(boosting_type = 'gbdt', objective='binary', learning_rate=0.1, max_bin = 510,
                     class_weight='balanced', n_estimators=250, num_leaves=20,
                                   max_depth=20,
                                   random_state=1, n_jobs=-1, metric='precision')

    lgbmF_clf.fit(combined_X, combined_Y)
    test_probs = lgbmF_clf.predict_proba(test_X)
    yPrime = [0 if x < optimal_threshold else 1 for x in test_probs[:,1]]
    print(accuracy_score(test_Y, yPrime))
    print(roc_auc_score(test_Y, test_probs[:,1]))
    print(classification_report(test_Y, yPrime))
    baseline_results.append({'model': 'tuned_lgbm_60min_windows',
                             'accuracy': accuracy_score(test_Y, yPrime),
                             'roc_auc': roc_auc_score(test_Y, test_probs[:,1]),
                             'precision': precision_score(test_Y, yPrime),
                             'recall': recall_score(test_Y, yPrime),
                             'f1-score': f1_score(test_Y, yPrime),
                             'j_index': optimal_threshold,
                             'pr_auc': average_precision_score(test_Y, test_probs[:,1]),
                             })

    os.makedirs('../data/output/training/', exist_ok=True)

    # Save top features
    pd.DataFrame({'feature': combined_X.columns,
                  'importance': lgbmF_clf.feature_importances_}).sort_values('importance', ascending = False).to_csv(
        '../data/output/training/lgbm_Final_featureImp_{}.csv'.format(datetime.today().strftime('%Y%m%d')))

    # Confusion Matrix
    cm = confusion_matrix(test_Y, yPrime)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig('../data/output/training/lgbm_Final_confMatrix_{}.png'.format(datetime.today().strftime('%Y%m%d')))
    plt.close()

    # Save the meal predictions
    test_df['predictions'] = yPrime
    test_df.to_csv('../data/output/training/lgbm_Final_predicitons_{}.csv'.format(
        datetime.today().strftime('%Y%m%d')), index = False)

    # Plot ROC Curve
    train_probs = lgbmF_clf.predict_proba(combined_X)
    optimal_threshold_train = j_index_threshold(combined_Y, train_probs[:,1], lgbm=True)
    yPrime_train = [0 if x < optimal_threshold else 1 for x in train_probs[:,1]]
    plot_roc_curves(combined_Y, train_probs[:,1], test_Y, test_probs[:,1])

    # Save results
    results = pd.DataFrame(baseline_results).sort_values('roc_auc', ascending=False)
    results.to_csv('../data/output/training/lgbm_Final_results.csv', index = False)
    print(results)

    # Save models
    os.makedirs('../data/output/models/', exist_ok=True)
    with open('../data/output/models/lgbm_mealDetection_model.pickle', 'wb') as handle:
        pickle.dump(lgbmF_clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # scalerfile = '../data/output/models/lgbm_standard_scaler.pickle'
    # pickle.dump(standScale, open(scalerfile, 'wb'))

    # Look at predictions
    for subj in test_df.subject.unique().tolist():
        plt.figure(figsize=(10,5))
        sub_df = test_df[test_df.subject == subj]
        plt.scatter(pd.to_datetime(sub_df.start_block), sub_df.meal, label = 'True Meals')
        plt.scatter(pd.to_datetime(sub_df.start_block), sub_df.predictions, color='red', label = 'predictions')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('No Meal <---> Meal')
        plt.savefig('../data/output/training/lgbm_Final_Preditions_{}_{}.png'.format(subj,
                                                                                 datetime.today().strftime('%Y%m%d')))
