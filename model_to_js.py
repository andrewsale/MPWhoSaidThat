# This converts a model produced by keras into a tensorflowjs model
# that can be loaded in run in a web browser.

import tensorflowjs as tfjs
import tensorflow as tf

model = tf.keras.models.load_model('Model-v5-6-0-0')

tfjs.converters.save_keras_model(model, 'Model_js')