import numpy as np
import pandas as pd
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

from keras.callbacks import EarlyStopping

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.keras import balanced_batch_generator
from imblearn.under_sampling import NearMiss

from src.keras_model import build_model, focal_loss
from src.processing import processData
from src.utils import pkl_save_obj

# Process Data

process_data = processData()

dfww = (process_data.get_events('data/jjWpmWpm_undecayed_01.csv')
        .append(process_data.get_events('data/jjWpmWpm_undecayed_02.csv'), ignore_index=True)
        .append(process_data.get_events('data/jjWpmWpm_undecayed_03.csv'), ignore_index=True))

# Classic Machine Learning

X = (dfww
     .drop('n_lon', axis = 1))
y = (dfww['n_lon'] == 2)

APs = dict()

## $\Delta\phi_{jj}$

scaler_phijj = StandardScaler()
X_phijj = (scaler_phijj
           .fit_transform(X['delta_phi.jj']
                          .values
                          .reshape(-1, 1)))

phijj_1 = LogisticRegression(solver='liblinear')

t1 = time()
cv_phijj_1 = cross_validate(phijj_1,
                            X_phijj,
                            y,
                            scoring='average_precision',
                            cv=5,
                            return_train_score=False)
print('Logistic Regression #1 finished in %0.1f seconds' %(time() - t1))

APs['phijj_default'] = cv_phijj_1['test_score']
print('AP of phijj_default = %0.3f +/- %0.3f' %(np.mean(APs['phijj_default']), np.std(APs['phijj_default'])))

phijj_2 = LogisticRegression(solver='liblinear',
                             class_weight='balanced')

t1 = time()
cv_phijj_2 = cross_validate(phijj_2,
                            X_phijj,
                            y,
                            scoring='average_precision',
                            cv=5,
                            return_train_score=False)
print('Logistic Regression #2 finished in %0.1f seconds' %(time() - t1))

APs['phijj_class_weight'] = cv_phijj_2['test_score']
print('AP of phijj_class_weight = %0.3f +/- %0.3f' %(np.mean(APs['phijj_class_weight']), np.std(APs['phijj_class_weight'])))

## Random Forest

rfc_1 = RandomForestClassifier(n_estimators=200,
                               max_depth=5)

t1 = time()
cv_rfc_1 = cross_validate(rfc_1,
                          X,
                          y,
                          scoring='average_precision',
                          cv=5,
                          return_train_score=False)
print('Random Forest #1 finished in %0.1f seconds' %(time() - t1))

APs['RF_default'] = cv_rfc_1['test_score']
print('AP of RF_default = %0.3f +/- %0.3f' %(np.mean(APs['RF_default']), np.std(APs['RF_default'])))

rfc_2 = RandomForestClassifier(n_estimators=200,
                               max_depth=5,
                               class_weight='balanced_subsample')

t1 = time()
cv_rfc_2 = cross_validate(rfc_2,
                          X,
                          y,
                          scoring='average_precision',
                          cv=5,
                          return_train_score=False)
print('Random Forest #2 finished in %0.1f seconds' %(time() - t1))

APs['RF_class_weight'] = cv_rfc_2['test_score']
print('AP of RF_class_weight = %0.3f +/- %0.3f' %(np.mean(APs['RF_class_weight']), np.std(APs['RF_class_weight'])))

rfc_3 = BalancedRandomForestClassifier(n_estimators=400,
                                       max_depth=5,
                                       replacement=True)

t1 = time()
cv_rfc_3 = cross_validate(rfc_3,
                          X,
                          y,
                          scoring='average_precision',
                          cv=5,
                          return_train_score=False)
print('Random Forest #3 finished in %0.1f seconds' %(time() - t1))

APs['RF_balanced'] = cv_rfc_3['test_score']
print('AP of RF_balanced = %0.3f +/- %0.3f' %(np.mean(APs['RF_balanced']), np.std(APs['RF_balanced'])))

pkl_save_obj(APs, 'APs')

# Deep Learning

X_tr, X_te, y_tr, y_te = train_test_split(X,
                                          y,
                                          test_size=0.2,
                                          stratify=y)
scaler_dnn = StandardScaler()
X_tr_dnn = (scaler_dnn
            .fit_transform(X_tr))
X_te_dnn = (scaler_dnn
            .transform(X_te))

APs_dnn = dict()

keras_model_1 = build_model()

t1 = time()
keras_model_1.fit(X_tr_dnn,
                  y_tr,
                  epochs=50,
                  batch_size=50,
                  verbose=1,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=20)],
                  validation_split=0.2)
print('Neural Network #1 finished in %0.1f seconds' %(time() - t1))

probas_dnn_1 = keras_model_1.predict_proba(X_te_dnn)

ap_dnn_1 = average_precision_score(y_te, probas_dnn_1)

APs_dnn['DNN_default'] = ap_dnn_1
print('AP of DNN_default = %0.3f' %APs_dnn['DNN_default'])

pkl_save_obj(APs_dnn, 'APs_dnn')

keras_model_1.save('results/keras_dnn_1.h5')

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y_tr),
                                                  y_tr)

keras_model_2 = build_model()

t1 = time()
keras_model_2.fit(X_tr_dnn,
                  y_tr,
                  epochs=50,
                  batch_size=50,
                  verbose=1,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=20)],
                  validation_split=0.2,
                  class_weight=class_weights)
print('Neural Network #2 finished in %0.1f seconds' %(time() - t1))

probas_dnn_2 = keras_model_2.predict_proba(X_te_dnn)

ap_dnn_2 = average_precision_score(y_te, probas_dnn_2)

APs_dnn['DNN_class_weight'] = ap_dnn_2
print('AP of DNN_class_weight = %0.3f' %APs_dnn['DNN_class_weight'])

pkl_save_obj(APs_dnn, 'APs_dnn')

keras_model_3 = build_model(loss_function=focal_loss())

t1 = time()
keras_model_3.fit(X_tr_dnn,
                  y_tr,
                  epochs=50,
                  batch_size=50,
                  verbose=1,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=20)],
                  validation_split=0.2)
print('Neural Network #3 finished in %0.1f seconds' %(time() - t1))

probas_dnn_3 = keras_model_3.predict_proba(X_te_dnn)

ap_dnn_3 = average_precision_score(y_te, probas_dnn_3)

APs_dnn['DNN_focal_loss'] = ap_dnn_3
print('AP of DNN_focal_loss = %0.3f' %APs_dnn['DNN_focal_loss'])

pkl_save_obj(APs_dnn, 'APs_dnn')

keras_model_4 = build_model()

training_generator, steps_per_epoch = balanced_batch_generator(X_tr_dnn,
                                                               y_tr,
                                                               sampler=NearMiss(),
                                                               batch_size=50)

t1 = time()
keras_model_4.fit_generator(generator=training_generator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=150,
                            verbose=1,
                            callbacks=[EarlyStopping(monitor='loss', patience=20)])
print('Neural Network #4 finished in %0.1f seconds' %(time() - t1))

probas_dnn_4 = keras_model_4.predict_proba(X_te_dnn)

ap_dnn_4 = average_precision_score(y_te, probas_dnn_4)

APs_dnn['DNN_balanced_batch'] = ap_dnn_4
print('AP of DNN_balanced_batch = %0.3f' %APs_dnn['DNN_balanced_batch'])

pkl_save_obj(APs_dnn, 'APs_dnn')
for clf, AP in APs_dnn.items():
    print('Average precision of ' + clf + ' = %0.3f' %AP)
