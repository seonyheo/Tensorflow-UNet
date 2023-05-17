import tensorflow as tf

def dice_loss(y_true, y_pred):
  y_pred = tf.sigmoid(y_pred)

  numerator = 2 * tf.math.reduce_sum(y_true * y_pred, axis=[1,2,3])
  denominator = tf.math.reduce_sum(y_true + y_pred, axis=[1,2,3])
  return tf.math.reduce_mean(1.0 - numerator / denominator)
