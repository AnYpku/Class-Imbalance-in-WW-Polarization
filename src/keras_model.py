from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential

from keras import backend as K
import tensorflow as tf

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss function from arXiv:1708.02002

    Keras implementation compatible w/ Tensorflow backend from
    https://github.com/mkocabas/focal-loss-keras
    """
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -(K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) +
                 K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0)))
    return focal_loss_fixed

def build_model(n_hidden_layers=2, loss_function='binary_crossentropy'):
    """
    deep, fully-connected neural network in Keras w/ Tensorflow backend

    takes 24 inputs and outputs a binary classification probability

    input and hidden layers have 150 nodes
    are initialized w/ He
    have relu activation functions
    and are regularized using dropout w/ a 50% rate and batch normalization

    output layer has a sigmoid activation function
    the adam optimizer is used
    """
    model = Sequential()

    model.add(Dense(150,
                    input_dim=24,
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    for _ in range(n_hidden_layers):
        model.add(Dense(150,
                        kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.compile(loss=loss_function,
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
