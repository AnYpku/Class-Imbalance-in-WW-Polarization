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
