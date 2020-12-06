import numpy as np
import cv2
import os
import pickle
from sklearn.model_selection import train_test_split
import pyprind

def prepare_dataset():
  dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/VariousImages"

  category2label = {}
  label2category = {}
  images = []
  labels = []

  for label, category in enumerate(os.listdir(dataset_path)):
    bar = pyprind.ProgBar(len(os.listdir(os.path.join(dataset_path, category))), track_time=True, title="Preparing {} images".format(category))

    for image_name in os.listdir(os.path.join(dataset_path, category)):
      image = cv2.imread(os.path.join(dataset_path, category, image_name))
      image = (cv2.resize(image, (100, 100) ) / 255.).astype(np.float32)
      images.append(image)
      labels.append(label)
      bar.update()

    category2label[category] = label
    label2category[label] = category

  images = np.array(images, dtype=np.float32)
  labels = np.array(labels, dtype=np.int32)
  x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, stratify=labels)

  with open(os.path.join(dataset_path, "pickle", "train_data.pkl"), "wb") as f:
    pickle.dump([x_train, y_train], f)

  with open(os.path.join(dataset_path, "pickle", "test_data.pkl"), "wb") as f:
    pickle.dump([x_test, y_test], f)

def load_test_dataset():
  dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/VariousImages"

  with open(os.path.join(dataset_path, "pickle", "test_data.pkl"), "rb") as f:
    x_test, y_test = pickle.load(f)

  return x_test, y_test

def load_train_dataset():
  dataset_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/VariousImages"

  with open(os.path.join(dataset_path, "pickle", "train_data.pkl"), "rb") as f:
    x_train, y_train = pickle.load(f)

  train_index = np.arange(0, y_train.shape[0])
  np.random.shuffle(train_index)

  return x_train, y_train
