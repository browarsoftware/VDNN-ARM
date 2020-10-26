#Order of execution: #17
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

def evaluate_arrays(a1, a2):

    K = set(np.asarray(a1))
    GT = set(np.asarray(a2))

    set_intersection = len(list(K.intersection(GT)))
    precision = set_intersection / len(list(K))
    # print(precision)

    recall = set_intersection / len(list(GT))
    # print(recall)

    accuracy = 0
    if set_intersection > 0:
        accuracy = 1
    # print(accuracy)
    return [precision, recall, accuracy]


ImageFile.LOAD_TRUNCATED_IMAGES = True


print('Loading data')
import numpy as np
Y = np.load('../Features/HashtagsLabels.bin.npy')


def PrepareModel(X, pathToWeights):
    model = keras.Sequential()
    model.add(Dense(2048, activation="relu", input_dim=X.shape[1]))
    model.add(Dense(2048, activation="relu"))
    #model.add(Dense(2048,activation='relu'))
    model.add(Dense(997,activation='sigmoid'))
    model.load_weights(pathToWeights)
    return model


models = []

weights = ['InceptionResNetV2.bin.npy', 'VGG16_Places365.bin.npy','InceptionV3.bin.npy']

pathsToWeights = ['InceptionResNetV2.bin.npy-155-0.36.hdf5', 'VGG16_Places365.bin.npy-147-0.37.hdf5','InceptionV3.bin.npy-170-0.36.hdf5']

#weights = ['DenseNet201_VGG16_Places365.bin.npy']
#pathsToWeights = ['DenseNet201_VGG16_Places365.bin.npy-91-0.36.hdf5']

id_ = 1
X = []
for id_ in range(len(weights)):
    print(id_)
    #X = np.load('../Hashtags50/Features/InceptionResNetV2_VGG16_Places365.bin.npy')
    X.append(np.load('../FeaturesBin/' + weights[id_]))
    pathToWeights = '../checkpointsPlaces/' + pathsToWeights[id_]
    models.append(PrepareModel(X[id_], pathToWeights))

import random
random.seed(4321)
my_random_sample = random.sample(range(X[0].shape[0]), 5000)

mask = np.ones(X[0].shape[0], dtype=bool)
mask[my_random_sample] = False

print("X = " + str(X[0].shape))
print("Y = " + str(Y.shape))

how_many_to_check = 5

#VALID
#print(my_random_sample)
for a in range(len(X)):
    X[a] = X[a][my_random_sample, ]

Y = Y[my_random_sample, ]
#print(X.shape)
#print(Y.shape)

how_much_data = X[0].shape[0]

precision1 = 0
recall1 = 0
accuracy1 = 0

precision5 = 0
recall5 = 0
accuracy5 = 0

res_array = np.zeros([Y.shape[0], how_many_to_check * len(pathsToWeights)])

################################
for my_id in range(Y.shape[0]):
    if my_id % 100 == 0:
        print(str(my_id) + " of " + str(Y.shape[0]))

    my_y = Y[my_id]
    my_y = np.nonzero(my_y)[0]

    prediction_OK = []
    for my_model_id in range(len(models)):
        my_x = X[my_model_id][my_id, :]
        my_x = my_x.reshape(1, my_x.shape[0])
        my_prediction = models[my_model_id].predict(my_x)
        my_prediction = my_prediction[0]
        my_prediction_sort = np.argsort(my_prediction)[::-1]

        #prediction_OK.append(my_prediction_sort[0:how_many_to_check])
        # print(evaluate_res)

        for a in range(how_many_to_check):
            aa = my_prediction_sort[0:how_many_to_check]
            res_array[my_id, (how_many_to_check * my_model_id) + a] = aa[a]
        #print("pred=" + str(my_prediction_sort[0:how_many_to_check]))
        #print("actual=" + str(my_y))
    print(res_array[my_id,:])

np.save('../Results/ensemble_results.bin', res_array)
