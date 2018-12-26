import numpy as np
import pandas as pd
from time import time

from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

from keras.callbacks import EarlyStopping
from keras.models import load_model

from imblearn.keras import balanced_batch_generator
from imblearn.under_sampling import NearMiss

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

X_tr, X_te, y_tr, y_te = train_test_split(X,
                                          y,
                                          test_size=0.2,
                                          stratify=y,
                                          random_state=4)
scaler_dnn = StandardScaler()
X_tr_dnn = (scaler_dnn
            .fit_transform(X_tr))
X_te_dnn = (scaler_dnn
            .transform(X_te))

# Deep Learning

APs_dnn = dict()

keras_model_1 = load_model('results/keras_model_50epochs.h5')
t1 = time()
keras_model_1.fit(X_tr_dnn,
                  y_tr,
                  epochs=200,
                  initial_epoch=50,
                  batch_size=50,
                  verbose=1,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=20)],
                  validation_split=0.2)
print('DNN finished in %0.1f seconds' %(time() - t1))
probas_dnn_1 = keras_model_1.predict_proba(X_te_dnn)
ap_dnn_1 = average_precision_score(y_te, probas_dnn_1)
APs_dnn['DNN'] = ap_dnn_1
print('DNN: AP = %0.3f' %APs_dnn['DNN'])
pkl_save_obj(APs_dnn, 'APs_dnn')
keras_model_1.save('results/keras_model_200epochs.h5')


class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(y_tr),
                                                  y_tr)
keras_model_2 = build_model()
t1 = time()
keras_model_2.fit(X_tr_dnn,
                  y_tr,
                  epochs=2,
                  batch_size=200,
                  verbose=1,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=20)],
                  validation_split=0.2,
                  class_weight=class_weights)
print('DNN w/ balanced weights finished in %0.1f seconds' %(time() - t1))
probas_dnn_2 = keras_model_2.predict_proba(X_te_dnn)
ap_dnn_2 = average_precision_score(y_te, probas_dnn_2)
APs_dnn['DNN w/ balanced weights'] = ap_dnn_2
print('DNN w/ balanced weights: AP = %0.3f' %APs_dnn['DNN w/ balanced weights'])
pkl_save_obj(APs_dnn, 'APs_dnn')

keras_model_3 = build_model(loss_function=focal_loss())
t1 = time()
keras_model_3.fit(X_tr_dnn,
                  y_tr,
                  epochs=2,
                  batch_size=200,
                  verbose=1,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=20)],
                  validation_split=0.2)
print('DNN w/ Focal Loss finished in %0.1f seconds' %(time() - t1))
probas_dnn_3 = keras_model_3.predict_proba(X_te_dnn)
ap_dnn_3 = average_precision_score(y_te, probas_dnn_3)
APs_dnn['DNN w/ Focal Loss'] = ap_dnn_3
print('DNN w/ Focal Loss AP: = %0.3f' %APs_dnn['DNN w/ Focal Loss'])
pkl_save_obj(APs_dnn, 'APs_dnn')

keras_model_4 = build_model()
t1 = time()
training_generator, steps_per_epoch = balanced_batch_generator(X_tr_dnn,
                                                               y_tr,
                                                               sampler=NearMiss(),
                                                               batch_size=50)
print('Balanced Batch Generation took %0.1f seconds' %(time() - t1))
t1 = time()
keras_model_4.fit_generator(generator=training_generator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=1300,
                            verbose=1,
                            callbacks=[EarlyStopping(monitor='loss', patience=20)])
print('DNN w/ balanced batches finished in %0.1f seconds' %(time() - t1))
probas_dnn_4 = keras_model_4.predict_proba(X_te_dnn)
ap_dnn_4 = average_precision_score(y_te, probas_dnn_4)
APs_dnn['DNN w/ balanced batches'] = ap_dnn_4
print('DNN w/ balanced batches: AP = %0.3f' %APs_dnn['DNN w/ balanced batches'])
pkl_save_obj(APs_dnn, 'APs_dnn')

for clf, AP in APs_dnn.items():
    print('Average precision of ' + clf + ' = %0.3f' %AP)
