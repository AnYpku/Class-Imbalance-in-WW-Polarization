import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from imblearn.ensemble import BalancedRandomForestClassifier

from src.analysis import classical, classification_report
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

# Classic Machine Learning
scores = dict()

## $\Delta\phi_{jj}$
print('Delta phi_jj')

X_phijj = (X['delta_phi.jj']
           .values
           .reshape(-1, 1))

phi_jj = LogisticRegression(solver='liblinear')

scores['Delta phi_jj'] = classical(phi_jj, X_phijj, y, skf)

classification_report(scores['Delta phi_jj'])

## Random Forest
print('Random Forest')

rfc_1 = RandomForestClassifier(n_estimators=500,
                               max_depth=10)

scores['Random Forest'] = classical(rfc_1, X.values, y, skf)

classification_report(scores['Random Forest'])

## Weighted Random Forest
print('Weighted Random Forest')

rfc_2 = RandomForestClassifier(n_estimators=500,
                               max_depth=10,
                               class_weight={0:1, 1:3})

scores['Weighted Random Forest'] = classical(rfc_2, X.values, y, skf)

classification_report(scores['Weighted Random Forest'])

## Balanced Random Forest
print('Balanced Random Forest')

rfc_3 = BalancedRandomForestClassifier(n_estimators=1000,
                                       replacement=True)

scores['Balanced Random Forest'] = classical(rfc_3, X.values, y, skf)

classification_report(scores['Balanced Random Forest'])

pkl_save_obj(scores, 'classical_scores')
