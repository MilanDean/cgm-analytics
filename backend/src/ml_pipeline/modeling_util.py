from sklearn.metrics import roc_curve, precision_recall_curve

import numpy as np
import pandas as pd

import random

# set seed
random.seed(1)
np.random.seed(1)

# warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)

def balance_onSubject(df):
    results = []
    for subject in df.subject.unique():
        sub_df = df[df.subject == subject]
        n_pos = sub_df[sub_df.meal == 1].shape[0]
        if n_pos < 1: print('{} has no meals'.format(subject))
        neg_df = sub_df[sub_df.meal == 0]
        neg_df_sampled = neg_df.sample(n_pos)
        features_balanced = pd.concat([sub_df[sub_df.meal == 1], neg_df_sampled], axis=0)
        results.append(features_balanced)

    return pd.concat(results)


def j_index_threshold(val_Y, val_probs, lgbm = False, metric = 'roc_auc'):
    '''
    Calculate the J-Index (optimal threshold for Binary labels)
    :param val_Y: True Y value
    :param val_probs: predicted value probability
    :param lgbm: flag to indicate if the model you are evaluating is an LGBM model
    :return: the threshold for binary classification (J-Index)
    '''
    if metric == 'roc_auc':
        if lgbm==True:
            fpr, tpr, _thresholds = roc_curve(val_Y, val_probs, pos_label=1)
        else:
            fpr, tpr, _thresholds = roc_curve(val_Y, val_probs[:, 1], pos_label=1)
    elif metric == 'pr_auc':
        if lgbm==True:
            fpr, tpr, _thresholds = precision_recall_curve(val_Y, val_probs, pos_label=1)
        else:
            fpr, tpr, _thresholds = precision_recall_curve(val_Y, val_probs[:, 1], pos_label=1)

    j_index = _thresholds[np.argmax(tpr - fpr)]
    return j_index
