import time
import pathlib
import tensorflow as tf
import numpy as np
from absl import flags
from absl import app

from models import unet
from data import carvana
from utils import loss

FLAGS = flags.FLAGS

flags.DEFINE_string('model', 'tinyunet', 'Model name')
flags.DEFINE_integer('batch', 16, 'Batch size')
flags.DEFINE_integer('epoch', 5, 'Number of epochs')

def main(argv):
  # Instantiate an optimizer.
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

  # Prepare the metrics.
  train_acc_metric = tf.keras.metrics.BinaryAccuracy()
  val_acc_metric = tf.keras.metrics.BinaryAccuracy()

  # Prepare the training dataset.
  train_dataset = carvana.CarvanaDataset("data/carvana/train.txt")
  train_dataset = train_dataset.shuffle(128).batch(FLAGS.batch)

  val_dataset = carvana.CarvanaDataset("data/carvana/val.txt")
  val_dataset = val_dataset.batch(FLAGS.batch)

  # Define a model.
  if FLAGS.model == 'tinyunet':
    model = unet.TinyUNet(classes=1)
  else:
    model = unet.UNet(classes=1)

  for epoch in range(FLAGS.epoch):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
      with tf.GradientTape() as tape:
        logits = model(x_batch_train, training=True)
        loss_value = loss.dice_loss(y_batch_train, logits)

      grads = tape.gradient(loss_value, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      # Update training metric.
      train_acc_metric.update_state(y_batch_train, logits)

      # Log every 200 batches.
      if step % 200 == 0:
        print("Training loss (for one batch) at step %d: %.4f"
          % (step, float(loss_value))
        )
        print("Seen so far: %d samples" % ((step + 1) * FLAGS.batch))

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch.
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
      val_logits = model(x_batch_val, training=False)
      # Update val metrics
      val_acc_metric.update_state(y_batch_val, val_logits)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))

  # Save the weights.
  model.save_weights("%s.h5" % FLAGS.model)

  # Save the tflite model.
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  tflite_model_file = pathlib.Path("%s.tflite" % FLAGS.model)
  tflite_model_file.write_bytes(tflite_model)

if __name__ == "__main__":
  app.run(main)
