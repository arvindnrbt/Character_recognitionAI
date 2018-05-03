import numpy as np
import tensorflow as tf
from skimage.io import imread
from PIL import Image
import os
import keras
from keras.models import Sequential
from keras.utils import *
from keras.layers import Dense, Conv2D, Flatten, Dropout
from skimage.transform import resize
from keras.preprocessing.image import img_to_array
import argparse

weight_path = './best_weights.hdf5'
img_rows=50
img_cols=50
num_classes = 52

label_dict = {
    0:'a',
    1:'b',
    2:'c',
    3:'d',
    4:'e',
    5:'f',
    6:'g',
    7:'h',
    8:'i',
    9:'j',
    10:'k',
    11:'l',
    12:'m',
    13:'n',
    14:'o',
    15:'p',
    16:'q',
    17:'r',
    18:'s',
    19:'t',
    20:'u',
    21:'v',
    22:'w',
    23:'x',
    24:'y',
    25:'z'
}

parser = argparse.ArgumentParser()
parser.add_argument("image")
args = parser.parse_args()
image_file = args.image

def pre_process_image_input(image_file):
    img = imread(image_file)
    # img = img
    img_array = np.array([img_to_array(img)])
    x = img_array /255
    return x

# CNN Architecture

character_model = Sequential()
character_model.add(Conv2D(
    30, kernel_size=(3,3), activation='relu',
    input_shape=(img_rows, img_cols, 3)
))
character_model.add(Conv2D(
    30, kernel_size=(3,3), activation='relu'))

character_model.add(Conv2D(
    30, kernel_size=(3,3),activation='relu'))
# character_model.add(Dropout(0.4))
character_model.add(Conv2D(
    30, kernel_size=(3,3),activation='relu'))
character_model.add(Dropout(0.4))

character_model.add(Conv2D(
    30, kernel_size=(3,3), activation='relu'))
character_model.add(Dropout(0.4))

character_model.add(Flatten())

character_model.add(Dense(512, activation='relu'))
character_model.add(Dense(num_classes, activation='softmax'))

character_model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer='adam',
            metrics=['accuracy'])

character_model.load_weights(weight_path)

x = pre_process_image_input(image_file)
predictions = character_model.predict_classes(x)
Image.fromarray(x).show()
print (label_dict[predictions[0]])
