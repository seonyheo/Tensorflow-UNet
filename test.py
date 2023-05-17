import tensorflow as tf
import numpy as np

from PIL import Image
from models import unet

# Define a model.
model = unet.UNet(classes=1)
model.build(input_shape=[1, 160, 240, 3])

# Load weights for the model.
model.load_weights('unet.h5')

img = Image.open('in.jpg')
img = np.array(img, dtype=float) / 255.0
img = tf.image.resize(img, [160, 240])
img = tf.expand_dims(img, axis=0)

logits = model(img).numpy()
logits = np.squeeze(1 / (1 + np.exp(- logits)), axis=0)
logits = np.matmul(logits, np.array([[255, 255, 255]]))

Image.fromarray(logits.astype(np.uint8)).save('out.jpg')
