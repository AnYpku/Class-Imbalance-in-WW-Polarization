import numpy as np
import pandas as pd
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from imblearn.combine import SMOTEENN
from imblearn.ensemble import BalancedRandomForestClassifier

from src.processing import processData
from src.utils import pkl_save_obj

# Process Data

process_data = processData()

dfww = (process_data.get_events('data/jjWpmWpm_undecayed_01.csv')
        .append(process_data.get_events('data/jjWpmWpm_undecayed_02.csv'), ignore_index=True)
        .append(process_data.get_events('data/jjWpmWpm_undecayed_03.csv'), ignore_index=True))

X = (dfww
     .drop('n_lon', axis = 1))
y = (dfww['n_lon'] == 2)

skf = StratifiedKFold(n_splits=5,
                      random_state=30)

scores = dict()

# Classic Machine Learning

def scores_cv(name, classifer, X, upsample=False):
    t1 = time()
    aps = []
    aucs = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if upsample:
            smote_enn = SMOTEENN()
            X_train, y_train = (smote_enn
                                .fit_resample(X_train, y_train))

        clf = classifer
        (clf
         .fit(X_train, y_train))

        probas = (clf
                  .predict_proba(X_test))
        aps.append(average_precision_score(y_test, probas.T[1]))
        aucs.append(roc_auc_score(y_test, probas.T[1]))

    print(name + ' finished in %0.1f seconds' %(time() - t1))

    scores[name] = {'average_precision':aps,
                    'roc_auc':aucs}
    print(name + ': AP = %0.3f +/- %0.3f' %(np.mean(aps), np.std(aps)))
    print(name + ': AUC = %0.3f +/- %0.3f' %(np.mean(aucs), np.std(aucs)))

## $\Delta\phi_{jj}$

X_phijj = (X['delta_phi.jj']
           .values
           .reshape(-1, 1))
phi_jj = LogisticRegression(solver='liblinear')
scores_cv('Delta phi_jj', phi_jj, X_phijj)

## Random Forest

rfc_1 = RandomForestClassifier(n_estimators=500,
                               max_depth=10)
scores_cv('Random Forest', rfc_1, X.values)

rfc_2 = RandomForestClassifier(n_estimators=500,
                               max_depth=10,
                               class_weight={0:1, 1:3})
scores_cv('Weighted Random Forest', rfc_2, X.values)

rfc_3 = BalancedRandomForestClassifier(n_estimators=1000,
                                       replacement=True)
scores_cv('Balanced Random Forest', rfc_3, X.values)

rfc_4 = RandomForestClassifier(n_estimators=500,
                               max_depth=10)
scores_cv('SMOTE ENN Random Forest', rfc_4, X.values, upsample=True)

pkl_save_obj(scores, 'scores')
