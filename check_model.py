import tensorflow as tf
from skimage.io import imread
import pandas as pd
import numpy as np

import keras
from keras.models import Sequential
from keras.utils import *
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator, img_to_array

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

df['prediction']=[label_dict[pred] for pred in predictions]

confusionMatrix = pd.crosstab(df['character'], df['prediction'], rownames=['Actual'], colnames=['Predicted'], margins=True)

confusionMatrix.to_csv('current_model_result.csv')

confusionMatrix = confusionMatrix.drop(labels='All', axis=1)
confusionMatrix = confusionMatrix.drop(labels='All', axis=0)

FP = confusionMatrix.sum(axis=0) - np.diag(confusionMatrix)  
FN = confusionMatrix.sum(axis=1) - np.diag(confusionMatrix)
TP = np.diag(confusionMatrix)
TN = confusionMatrix.values.sum() - (FP + FN + TP)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)

# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print ('True Positive',TPR)
print ('False Negative',FNR)
print ('False Positive',FPR)
print ('True Negative',TNR)

print ('Accuracy', ACC)
