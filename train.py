import numpy as np
import pandas as pd
import keras
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from skimage.io import imread
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.utils import *
import tensorflow as tf
import os
# np.set_printoptions(threshold=np.nan)

model_save_path = './best_weights.hdf5'
img_rows = 28
img_cols = 28

num_train_set = 0

num_classes = 26
FP = os.getcwd()

image_dir = 'Images'
input_path = os.path.join(FP,image_dir)
data_file = pd.read_csv('Train.csv')

def pre_process_image_input(data):
    labels = data['label'].tolist()
    onehot_y = to_categorical(labels, num_classes)
    num_train_set = len(data)

    image_files = [os.path.join(image_dir, img) for img in data['image'].tolist()]
    imgs = [imread(img_path) for img_path in image_files]
    img_array = np.array([(img_to_array(img)) for img in imgs])
    # x = preprocess_input(img_array)
    x = img_array /255
    x_train, x_test, y_train, y_test = train_test_split(x,onehot_y)
    return x_train, x_test, y_train, y_test

data_gen = ImageDataGenerator(
# featurewise_center=True, #
# zca_whitening=True,#
width_shift_range = 0.2,
height_shift_range = 0.2,
rescale=2
)

train_input, test_input, train_label, test_label = pre_process_image_input(data_file)

train_generator = data_gen.flow(
    x=train_input,
    y=train_label,
    seed=4
)

test_generator = data_gen.flow(
    x=test_input,
    y=test_label,
    seed=4
)

earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
modelCheckpoint = keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

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
character_model.summary()

character_model.fit_generator(train_generator,
          epochs=20,
        #   steps_per_epoch = 20,
          validation_data = test_generator,
          callbacks=[earlyStopping, modelCheckpoint])

score = character_model.evaluate(test_input, test_label, batch_size=128)
print ('score',score)
