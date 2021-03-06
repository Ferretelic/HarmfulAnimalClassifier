from animal import load_test_dataset
from keras.utils import to_categorical
from keras.models import load_model
import tensorflow as tf
import numpy as np
import pandas as pd

physical_devices = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def create_submission_file(epoch):
  x_test, y_test = load_test_dataset()

  model = load_model("./models/model_{}.h5".format(epoch))
  predictions = np.argmax(model.predict(x_test), axis=1)
