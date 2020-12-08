import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import cv2 as cv
import os
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils.np_utils import to_categorical
from src import plot_history as ph
from src import prep_data as pd

## Create Model
def train(train_images, test_images, train_labels, test_labels, input_res, num_classes):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, activation='relu',
                     input_shape=(input_res[0], input_res[1], 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    ## Save weights
    model.summary()
    callbacks_list = []
    filepath = "sportsballs_7c_unreg.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list.append(checkpoint)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    callbacks_list.append(tensorboard_callback)
    ## Train/Validate Model
    history = model.fit(train_images, train_labels, batch_size=64, epochs=75,
                            validation_data=(test_images, test_labels), callbacks=callbacks_list)

    # plot training session
    ph.plot_acc_loss(history, "Unregularized 7-Class Model")
    return model, history

def load_weights():
    # load weights into new model
    loaded_model = load_model("sportsballs_7c_unreg.hdf5")
    print("Loaded model from disk")
    loaded_model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return loaded_model

## Testing
try:
    model, history = train(train_images, test_images, train_labels, test_labels, input_res, num_classes)
except NameError:
    train_images, test_images, train_labels, test_labels, input_res, num_classes, dictionary = pd.load_data()
    model, history = train(train_images, test_images, train_labels, test_labels, input_res, num_classes)
