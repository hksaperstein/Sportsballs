import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

from tensorflow.python.keras.utils.np_utils import to_categorical

import plot_history as ph
from sklearn import model_selection
import gpu_mem_fix

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

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.33)
## Create Model

model = Sequential()
model.add(Conv2D(8, kernel_size=3, padding='same', activation='relu',
                 input_shape=(input_res[0], input_res[1], 3), kernel_constraint=max_norm(4)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(512, activation='relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

## Save weights
model.summary()
callbacks_list = []
filepath = "sportsballs_classification_weights2.hdf5"
# filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list.append(checkpoint)

## Train/Validate Model
history = model.fit(train_images, train_labels, batch_size=64, epochs=50,
                        validation_data=(test_images, test_labels), callbacks=callbacks_list)

# plot training session
ph.plot_acc_loss(history, "Regularized Model")