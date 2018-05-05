import numpy as np
import tensorflow as tf
from skimage.io import imread
from PIL import Image
import os, string
import keras
from keras.models import Sequential
from keras.utils import *
from keras.layers import Dense, Conv2D, Flatten, Dropout
from skimage.transform import resize
from keras.preprocessing.image import img_to_array
import argparse
from configuration import config

# Constant
weight_path = './best_weights.hdf5'
img_rows=config['IMAGE_ROWS']
img_cols=config['IMAGE_COLUMNS']
num_classes = config['NUM_CLASSES']
num_kernels = config['NUM_FILTERS']

# Argument
parser = argparse.ArgumentParser()
parser.add_argument("image")
args = parser.parse_args()
image_file = args.image

# Ascii letters
label_vals = list(string.ascii_letters)

# Pre-Process image
def pre_process_image_input(image_file):
    img = imread(image_file)
    img_array = np.array([img_to_array(img)])
    x = img_array /255
    return x

# ----------- CNN ARCHITECTURE -------------

character_model = Sequential()
character_model.add(Conv2D(
    num_kernels, kernel_size=(3,3), activation='relu',
    input_shape=(img_rows, img_cols, 3)
))
character_model.add(Conv2D(
    num_kernels, kernel_size=(3,3), activation='relu'))
character_model.add(Conv2D(
    num_kernels, kernel_size=(3,3),activation='relu'))

character_model.add(Conv2D(
    num_kernels, kernel_size=(3,3),activation='relu'))
character_model.add(Dropout(0.4))

character_model.add(Conv2D(
    num_kernels, kernel_size=(3,3), activation='relu'))
character_model.add(Conv2D(
    num_kernels, kernel_size=(3,3), activation='relu'))

character_model.add(Dropout(0.4))


character_model.add(Flatten())

character_model.add(Dense(512, activation='relu'))
character_model.add(Dense(num_classes, activation='softmax'))

character_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

# ----------- CNN ARCHITECTURE -------------

character_model.load_weights(weight_path)

x = pre_process_image_input(image_file)
predictions = character_model.predict_classes(x)
print (label_dict[predictions[0]])
