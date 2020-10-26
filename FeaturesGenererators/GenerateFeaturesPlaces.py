#Order of execution: #3
# https://github.com/GKalliatakis/Keras-Application-Zoo
import os
#import urllib2
import urllib.request as urllib2
import numpy as np
from PIL import Image
from cv2 import resize
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
from vgg16_places_365 import VGG16_Places365
import pandas as pd

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

def my_preprocess_input(TEST_IMAGE_PATH):
    image = Image.open(TEST_IMAGE_PATH)
    image = np.array(image, dtype=np.uint8)
    image = resize(image, (224, 224))
    if len(image.shape) < 3:
        images_copy = np.zeros([224, 224, 3])
        images_copy[:,:,0] = image.copy()
        images_copy[:, :, 1] = image.copy()
        images_copy[:, :, 2] = image.copy()
        image = images_copy

    image = np.expand_dims(image, 0)
    return image

TEST_IMAGE_URL = 'http://places2.csail.mit.edu/imgs/demo/6.jpg'

image = Image.open(urllib2.urlopen(TEST_IMAGE_URL))
image = np.array(image, dtype=np.uint8)
image = resize(image, (224, 224))
image = np.expand_dims(image, 0)

base_model = VGG16_Places365(weights='places', include_top=False)
# print(models_name[model_id])
x = base_model.output
x = GlobalAveragePooling2D()(x)
model = Model(inputs=base_model.input, outputs=x)

path_to_data = 'd:\\dane\\HARRISON\\'
csv_ok = pd.read_csv('d:\\dane\\HARRISON\\data_list.txt', header=None)

models_name = ['VGG16_Places365']

model_id = 0

#for a in range(csv_ok.shape[0]):
for a in range(csv_ok.shape[0]):
    if a % 100 == 0:
        print(str(a) + " of " + str(csv_ok.shape[0]))
    my_file = str(csv_ok.iloc[a, 0])
    #print(my_file)
    img_path = path_to_data + my_file
    x = my_preprocess_input(img_path)
    #if x.shape[1] != 224:
    #print(x.shape)
    features = model.predict(x)

    features = features[0]
    #print(features.shape)
    #print(features)

    file_object = open('../Features/' + models_name[model_id] + 'Features.txt', 'a')
    for b in range(features.shape[0]):
        if b > 0:
            file_object.write(",")
        file_object.write(str(features[b]))
    file_object.write('\n')
    file_object.close()
