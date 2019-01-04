import numpy as np
from time import time

from sklearn.metrics import average_precision_score, roc_auc_score

from src.compute_LL import logLikelihood

log_like = logLikelihood()

def classical(classifer, X, y, folder):
    times = []
    aps = []
    aucs = []
    sigmas = []

    for train_index, test_index in folder.split(X, y):
        t0 = time()

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = classifer
        (clf
         .fit(X_train, y_train))

        probas = (clf
                  .predict_proba(X_test))
        aps.append(average_precision_score(y_test, probas.T[1]))
        aucs.append(roc_auc_score(y_test, probas.T[1]))

        mLL = (log_like.compute_log_likelihood(clf, X_test, y_test) /
               log_like.rescale)
        sigmas.append(log_like.compute_sigma(mLL))

        times.append(time() - t0)

    return {'times':times,
            'average_precision':aps,
            'roc_auc':aucs,
            'sigmas':sigmas}

def classification_report(scores):
    print('Time / Fold = %0.1f +/- %0.1f s' %(np.mean(scores['times']),
                                              np.std(scores['times'])))
    print('Average Precision = %0.3f +/- %0.3f' %(np.mean(scores['average_precision']),
                                                  np.std(scores['average_precision'])))
    print('ROC AUC = %0.3f +/- %0.3f' %(np.mean(scores['roc_auc']),
                                        np.std(scores['roc_auc'])))
    print('Significance = %0.1f +/- %0.1f' %(np.mean(scores['sigmas']),
                                             np.std(scores['sigmas'])))
