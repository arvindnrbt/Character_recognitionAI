import tensorflow as tf
from skimage.io import imread
import pandas as pd
import numpy as np
import string
import keras
from keras.models import Sequential
from keras.utils import *
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import os
from configuration import config

# Constants
IMG_PATH = './'+config['IMAGE_PATH']+'/'
weight_path = './best_weights.hdf5'
FONT_PATH = './'+config['FONT_PATH']+'/'
Train_file = config['TRAIN_CSV']

num_classes = config['NUM_CLASSES']
img_rows = config['IMAGE_ROWS']
img_cols = config['IMAGE_COLUMNS']
num_kernels = config['NUM_FILTERS']

CONFUSION_MATRIX_CSV = './Run/current_model_result.csv'
RESULT_CSV = './Run/Result.csv'
METRICS_CSV = './Run/Metrics.csv'

# Preparing list of Upper and Lower case alphabets
label_vals = list(string.ascii_letters)[0:26] + list(string.digits)

# Preprocess function
def pre_process_image_input(image_files, labels):
    imgs = [imread(img_path) for img_path in image_files]
    img_array = np.array([(img_to_array(img)) for img in imgs])
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
# Max pooling
character_model.add(MaxPooling2D(pool_size=(3,3)))

character_model.add(Conv2D(
    num_kernels, kernel_size=(3,3), activation='relu'))
character_model.add(Conv2D(
    num_kernels, kernel_size=(3,3), activation='relu'))

character_model.add(Flatten())

character_model.add(Dense(512, activation='relu'))

# Dropout
character_model.add(Dropout(0.2))

character_model.add(Dense(num_classes, activation='softmax'))

character_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])


# ----------- CNN ARCHITECTURE -------------

# Load weights
character_model.load_weights(weight_path)

# Read CSV
df = pd.read_csv(Train_file)

# Eliminate non-existant files
# for index, img in enumerate(df['image'].tolist()):
#     if not os.path.exists(img):
#         df.drop(index,inplace=True)


labels = df['label'].tolist()
image_files = df['image'].tolist()

# Preprocess to get x array and y label
x = pre_process_image_input(image_files,labels)

predictions = character_model.predict_classes(x)

df['prediction']=[label_vals[pred] for pred in predictions]

df.to_csv(RESULT_CSV)

# Prepare ConfusionMatrix
confusionMatrix = pd.crosstab(df['identifier'], df['prediction'], rownames=['Actual'], colnames=['Predicted'], margins=True)

# Write ConfusionMatrix to CSV
confusionMatrix.to_csv(CONFUSION_MATRIX_CSV)

# Drop All label for calculating TPR, FPR, etc.
confusionMatrix = confusionMatrix.drop(labels='All', axis=1)
confusionMatrix = confusionMatrix.drop(labels='All', axis=0)

# Calculate Metrics
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

# print ('True Positive',TPR)
# print ('False Negative',FNR)
# print ('False Positive',FPR)
# print ('True Negative',TNR)

# print ('Accuracy', ACC)

dataF = {
    'True Positive' : TPR,
    'False Positive': FPR,
    'True Negative': TNR,
    'False Negative': FNR,
    'Accuracy': ACC
}

Metrics = pd.DataFrame(data=dataF)
print (Metrics)

Metrics.to_csv(METRICS_CSV)
