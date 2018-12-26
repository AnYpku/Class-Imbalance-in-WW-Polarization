import numpy as np
import pandas as pd
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

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

APs = dict()

# Classic Machine Learning

def ap_cv(name, clf, X):
    t1 = time()
    cv = cross_validate(clf,
                        X,
                        y,
                        scoring='average_precision',
                        cv=5,
                        return_train_score=False)
    print(name + ' finished in %0.1f seconds' %(time() - t1))

    APs[name] = cv['test_score']
    print(name + ': AP = %0.3f +/- %0.3f' %(np.mean(APs[name]), np.std(APs[name])))

## $\Delta\phi_{jj}$

scaler_phijj = StandardScaler()
X_phijj = (scaler_phijj
           .fit_transform(X['delta_phi.jj']
                          .values
                          .reshape(-1, 1)))
phijj_1 = LogisticRegression(solver='liblinear')
ap_cv('Delta phi_jj', phijj_1, X_phijj)


phijj_2 = LogisticRegression(solver='liblinear',
                             class_weight='balanced')
ap_cv('Delta phi_jj w/ balanced weights', phijj_2, X_phijj)

## Random Forest

rfc_1 = RandomForestClassifier(n_estimators=200,
                               max_depth=5)
ap_cv('Random Forest', rfc_1, X)

rfc_2 = RandomForestClassifier(n_estimators=200,
                               max_depth=5,
                               class_weight='balanced_subsample')
ap_cv('Random Forest w/ balanced weights', rfc_2, X)

rfc_3 = BalancedRandomForestClassifier(n_estimators=1000,
                                       max_depth=5,
                                       replacement=True)
ap_cv('Balanced Random Forest', rfc_3, X)

pkl_save_obj(APs, 'APs')
