import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from skimage.io import imread
import numpy as np
from keras.models import Sequential
from keras.utils import *
from keras.layers import Dense, Conv2D, Flatten, Dropout
import os

weight_path = './best_weights.hdf5'
num_classes = 26
img_rows=28
img_cols=28
num_fonts = 332

# image_files = [
#     './Images/a0.png',
#     './Images/b313.png',
#     './Images/c646.png',
#     './Images/d939.png',
#     './Images/e1252.png',
#     './Images/f1565.png',
#     './Images/g1878.png',
#     './Images/h2191.png',
#     './Images/i2514.png',
#     './Images/j2817.png',
#     './Images/k3130.png',
#     './Images/l3443.png',
#     './Images/m3756.png',
#     './Images/n4069.png',
#     './Images/o4382.png',
#     './Images/p4695.png',
#     './Images/q5008.png',
#     './Images/r5321.png',
#     './Images/s5634.png',
#     './Images/t5947.png',
#     './Images/u6260.png',
#     './Images/v6573.png',
#     './Images/w6886.png',
#     './Images/x7199.png',
#     './Images/y7512.png',
#     './Images/z7825.png',    
# ]

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

def pre_process_image_input(image_files):
    labels = list(label_dict.keys()) #[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    onehot_y = to_categorical(labels, num_classes)
    imgs = [imread(img_path) for img_path in image_files]
    img_array = np.array([(img_to_array(img)) for img in imgs])
    # x = preprocess_input(img_array)
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

x = 1
while (x is not '0'):
    try:
        char = str(input('Enter the character to find results: '))
    except ValueError:
        continue
    index = label_vals.index(char)

    st = num_fonts*index
    ed = st+num_fonts

    image_files = []
    for i in range(st,ed):
        image_files.append('./Images/'+char+str(i)+'.png')

    x,y = pre_process_image_input(image_files)

    predictions = character_model.predict_classes(x)

    count =0
    for ind,pred in enumerate(predictions):
        # print(label_dict[pred])
        if label_dict[pred] == char:
            count = count+1

    print (str(count)+' / '+str(num_fonts))
    x = input("\n\tEnter 0 to quit..")
