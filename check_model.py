import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from skimage.io import imread
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.utils import *
from keras.layers import Dense, Conv2D, Flatten, Dropout
import os

IMG_PATH = './Images/'
weight_path = './best_weights.hdf5'
FONT_PATH = './Font Pack/'
train_csv = 'Train.csv'
num_classes = 26
img_rows=28
img_cols=28
num_fonts = 196
offsets = 6
num_set = num_fonts * offsets

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
st = 0
label_vals = list(label_dict.values())

def pre_process_image_input(image_files, labels):
    onehot_y = to_categorical(labels, num_classes)
    imgs = [imread(img_path) for img_path in image_files]
    img_array = np.array([(img_to_array(img)) for img in imgs])
    x = img_array /255
    return x, onehot_y

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

# Prediction
df = pd.read_csv(train_csv)

labels = df['label'].tolist()
image_files = df['image'].tolist()

x,y = pre_process_image_input(image_files,labels)

predictions = character_model.predict_classes(x)

df.assign(prediction=predictions)
   
print ('True Positive')
print ('True Negative')
print ('False Positive')
print ('False Negative')
