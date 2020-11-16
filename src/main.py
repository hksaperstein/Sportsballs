import tensorflow as tf
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from sklearn import model_selection
## Load Data

#todo use sys dir path stuff
data_path = "/home/hksaperstein/WPI/rbe549/Project/Sportsballs/data/"
sportballs = ["soccerballs/", "baseballs/"]

image_paths = []
num_images = 0
for i, sportball in enumerate(sportballs):
    image_files = os.listdir("/home/hksaperstein/WPI/rbe549/Project/Sportsballs/data/" + sportball)
    if 'annotations' in image_files:
        image_files.remove('annotations')
    num_images += len(image_files)
    image_paths.append(image_files)

images = np.empty((num_images, 100, 100, 3), dtype='uint8')
labels = np.empty((num_images,), dtype='uint8')
## todo array of labels based off balls
soccer_label = 0
image = 0
# Load data
current_num_images = 0
for i, image_path in enumerate(image_paths):
    for j, image_file in enumerate(image_path):
        raw_image = cv.imread(data_path + sportballs[i] + image_file)
        formatted_image = cv.resize(raw_image, (100, 100))
        # print(type(formatted_image))
        images[j + current_num_images] = formatted_image
        labels[j + current_num_images] = i
    current_num_images += len(image_path)
#split data
train_images, test_images, train_labels, test_labels = model_selection.train_test_split(images, labels, test_size=.2)

## Create Model

model = Sequential()
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(100, 100, 3), kernel_constraint=max_norm(4)))
model.add(Dropout(0.1))
# model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.1))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.1))
# model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.1))
# model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(10, kernel_size=3, padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
# model.add(Dropout(0.2))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))
# model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.4))
# model.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.5))
# model.add(Dense(2048, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.5))
model.add(Dense(10))
# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
## Train/Validate Model
history = model.fit(train_images, train_labels, batch_size=64, epochs=100,
                        validation_data=(test_images, test_labels))


# plot training session
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
axes = plt.gca()
axes.set_ylim([0, 1])
plt.show()
