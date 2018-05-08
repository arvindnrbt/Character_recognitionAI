import numpy as np
import pandas as pd
import keras
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from skimage.io import imread
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.utils import *
import tensorflow as tf
import os
from configuration import config
# np.set_printoptions(threshold=np.nan)

# Constants
model_save_path = './best_weights.hdf5'

num_classes = config['NUM_CLASSES']
img_rows = config['IMAGE_ROWS']
img_cols = config['IMAGE_COLUMNS']
num_kernels = config['NUM_FILTERS']
Train_file = config['TRAIN_CSV']

# Read file
data_file = pd.read_csv(Train_file)

# Split data into 60%, 20%, 10% for train, val, test
def split_train_data(x):
    train, val, test = np.split(x, [int(.6*len(x)), int(.8*len(x))])
    return train, val, test

# Pre-Processing
def pre_process_image_input(data):
    labels = data['label'].tolist()
    onehot_y = to_categorical(labels, num_classes)
    image_files = data['image'].tolist()
    imgs = [imread(img_path) for img_path in image_files]
    img_array = np.array([(img_to_array(img)) for img in imgs])
    x = img_array /255
    x_train, x_val, x_test = split_train_data(x)
    y_train, y_val, y_test = split_train_data(onehot_y)
    return x_train, x_val, x_test, y_train, y_val, y_test

# Image data generator for data augmentation
data_gen = ImageDataGenerator(
width_shift_range = 0.2,
height_shift_range = 0.2,
rescale=2
)
# Eliminate Non existant files
# for index, img in enumerate(data_file['image'].tolist()):
#     if not os.path.exists(img):
#         data_file.drop(index,inplace=True)

# Shuffling
data_file = data_file.sample(frac=1).reset_index(drop=True)

# Pre-Processing image and split into train, val and test
train_input, val_input, test_input, train_label, val_label, test_label = pre_process_image_input(data_file)

# Train generator
train_generator = data_gen.flow(
    x=train_input,
    y=train_label,
    seed=4
)

# Val generator
val_generator = data_gen.flow(
    x=val_input,
    y=val_label,
    seed=4
)

# Configure EarlyStopping if the model is Overfitting
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')

# Save model every time loss gets lesser for this run 
modelCheckpoint = keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

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


# Display summary
character_model.summary()

# Fit model
character_model.fit_generator(train_generator,
          epochs=8,
        #   steps_per_epoch = 20,
          validation_data = val_generator,
          callbacks=[earlyStopping, modelCheckpoint])

# Evaluation on test data which is 10% split (Unseen) from training data
score = character_model.evaluate(test_input, test_label, batch_size=512)
print ('score',score)
