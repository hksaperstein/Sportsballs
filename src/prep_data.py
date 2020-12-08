import os
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical

# Loads only the first three classes from /data/
def load_data3c():
    seed = 7
    np.random.seed(seed)

    ## Load Data
    data_path = "data"
    names = os.listdir(data_path)
    input_res = (128, 128)
    # load data from folders based on folder name:
    images = []
    labels = []
    label_dict = {}
    num_images = 0
    for i, name in enumerate(names[0:3]):
        # count examples
        image_files = os.listdir(data_path + '/' + name)
        num_images += len(image_files)
        im_path = data_path + '/' + names[0]
        # encoded category for name
        label_dict.update({name: i})

        # load image into example pool
        for im_name in image_files:
            im = data_path + '/' + name + '/' + im_name
            im = cv.imread(im)
            image = cv.resize(im, dsize=(input_res))
            images.append(image)
            labels.append(i)

    # Normalize pixel values to be between 0 and 1
    images = np.asarray(images)
    labels = to_categorical(np.asarray(labels))
    images = images / 255.0
    num_classes = len(labels[0])

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)
    return train_images, test_images, train_labels, test_labels, input_res, num_classes, label_dict

# Loads all folders from /data/
def load_data():
    seed = 7
    np.random.seed(seed)

    ## Load Data
    data_path = "data"
    names = os.listdir(data_path)
    input_res = (128, 128)
    # load data from folders based on folder name:
    images = []
    labels = []
    label_dict = {}
    num_images = 0
    for i, name in enumerate(names):
        # count examples
        image_files = os.listdir(data_path + '/' + name)
        num_images += len(image_files)
        im_path = data_path + '/' + names[0]
        # encoded category for name
        label_dict.update({name: i})

        # load image into example pool
        for im_name in image_files:
            im = data_path + '/' + name + '/' + im_name
            im = cv.imread(im)
            image = cv.resize(im, dsize=(input_res))
            images.append(image)
            labels.append(i)

    # Normalize pixel values to be between 0 and 1
    images = np.asarray(images)
    labels = to_categorical(np.asarray(labels))
    images = images / 255.0
    num_classes = len(labels[0])

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)
    return train_images, test_images, train_labels, test_labels, input_res, num_classes, label_dict
