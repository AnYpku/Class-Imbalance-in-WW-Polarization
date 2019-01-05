import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from keras.callbacks import EarlyStopping

from imblearn.keras import balanced_batch_generator

from src.analysis import deep, classification_report
from src.focal_loss import focal_loss
from src.keras_model import build_model
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

scaler_dnn = StandardScaler()
X_dnn = (scaler_dnn
         .fit_transform(X))

skf = StratifiedKFold(n_splits=5,
                      random_state=30)

# Deep Learning
early_stopping = EarlyStopping(monitor='loss',
                               patience=10)
scores = dict()

## DNN
dnn_1 = build_model()

scores['DNN'] = deep(dnn_1,
                     X_dnn,
                     y,
                     skf,
                     early_stopping)

dnn_1.save('results/dnn.h5')

classification_report(scores['DNN'])

pkl_save_obj(scores, 'deep_scores') # save early and save often

## Weighted DNN - use Focal Loss to implement weights
dnn_2 = build_model(loss_function=focal_loss(gamma=0.0, alpha=0.75))

scores['Weighted DNN'] = deep(dnn_2,
                              X_dnn,
                              y,
                              skf,
                              early_stopping)

classification_report(scores['Weighted DNN'])

pkl_save_obj(scores, 'deep_scores')

## DNN w/ Focal Loss
dnn_3 = build_model(loss_function=focal_loss())

scores['DNN w/ Focal Loss g=2.0, a=0.25'] = deep(dnn_3,
                                                 X_dnn,
                                                 y,
                                                 skf,
                                                 early_stopping)

classification_report(scores['DNN w/ Focal Loss g=2.0, a=0.25'])

pkl_save_obj(scores, 'deep_scores')

## DNN w/ Focal Loss
dnn_4 = build_model(loss_function=focal_loss(gamma=0.5, alpha=0.5))

scores['DNN w/ Focal Loss g=0.5, a=0.5'] = deep(dnn_4,
                                                X_dnn,
                                                y,
                                                skf,
                                                early_stopping)

classification_report(scores['DNN w/ Focal Loss g=0.5, a=0.5'])

pkl_save_obj(scores, 'deep_scores')

## DNN w/ Focal Loss
dnn_5 = build_model(loss_function=focal_loss(gamma=0.2, alpha=0.75))

scores['DNN w/ Focal Loss g=0.2, a=0.75'] = deep(dnn_5,
                                                 X_dnn,
                                                 y,
                                                 skf,
                                                 early_stopping)

classification_report(scores['DNN w/ Focal Loss g=0.2, a=0.75'])

pkl_save_obj(scores, 'deep_scores')


## Balanced Batch DNN
dnn_6 = build_model()

scores['Balanced Batch DNN'] = deep(dnn_6,
                                    X_dnn,
                                    y,
                                    skf,
                                    early_stopping,
                                    generator=balanced_batch_generator)

classification_report(scores['Balanced Batch DNN'])

pkl_save_obj(scores, 'deep_scores')
