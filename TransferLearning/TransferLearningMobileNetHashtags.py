#Order of execution: #5
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

'''
DataFile = 'Xception.bin.npy'
DataFile = 'DenseNet201.bin.npy'
DataFile = 'InceptionResNetV2.bin.npy'
DataFile = 'VGG16_Places365.bin.npy'
DataFile = 'NASNetLarge.bin.npy'
DataFile = 'InceptionV3.bin.npy'
DataFile = 'MobileNetV2.bin.npy'
'''
DataFiles = ['Xception.bin.npy', 'DenseNet201.bin.npy', 'InceptionResNetV2.bin.npy', 'VGG16_Places365.bin.npy',
             'NASNetLarge.bin.npy', 'InceptionV3.bin.npy', 'MobileNetV2.bin.npy']

for DataFile in DataFiles:
    X = np.load('../FeaturesBin/' + DataFile)

    model = keras.Sequential()


    model.add(Dense(2048, activation="relu", input_dim=X.shape[1]))
    model.add(Dense(2048, activation="relu"))
    #model.add(Dense(2048,activation='relu'))
    model.add(Dense(997,activation='sigmoid'))

    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
    batch_size = 64
    step_size_train=X.shape[0]//batch_size

    import numpy as np
    Y = np.load('../Features/HashtagsLabels.bin.npy')

    import random
    random.seed(4321)
    my_random_sample = random.sample(range(X.shape[0]), 5000)

    mask = np.ones(X.shape[0], dtype=bool)
    mask[my_random_sample] = False

    print("X = " + str(X.shape))
    print("Y = " + str(Y.shape))

    #TRAIN
    X = X[mask, ]
    Y = Y[mask, ]
    print(X.shape)
    print(Y.shape)

    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import LearningRateScheduler
    # checkpoint
    filepath="../CheckpointsPlaces/" + DataFile + "-{epoch:02d}-{accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

    def lr_scheduler(epoch, lr):
        #if epoch == 1:
        #    lr = 0.01
        if epoch % 200 == 0 and epoch > 0:
            lr = lr * 0.1
        return lr

    callbacks_list = [checkpoint,LearningRateScheduler(lr_scheduler, verbose=1)]

    model.fit(x = X,
              y = Y,
              shuffle=True,
              batch_size = batch_size,
              steps_per_epoch=step_size_train,
              epochs=200,
              callbacks=callbacks_list)

