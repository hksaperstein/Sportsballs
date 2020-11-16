import tensorflow as tf
import numpy as np
import cv2 as cv
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

train_images = np.empty((num_images, 3, 100, 100), dtype='uint8')
train_labels = np.empty((num_images,), dtype='uint8')
## todo array of labels based off balls
soccer_label = 0
image = 0
# Load data
current_num_images = 0
for i, image_path in enumerate(image_paths):
    print(i)
    for j, image_file in enumerate(image_path):
        raw_image = cv.imread(data_path + sportballs[i] + image_file)
        formatted_image = np.resize(raw_image, (3, 100, 100))
        # print(type(formatted_image))
        train_images[j + current_num_images] = formatted_image
        train_labels[j + current_num_images] = i
    current_num_images += len(image_path)
#split data
# model_selection.train_test_split()
## Create Model


## Train/Validate Model