#Order of execution: #12
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from keras.applications.resnet50 import ResNet50

from keras.applications.inception_v3 import preprocess_input
from keras.applications.inception_v3 import decode_predictions
from keras.applications.inception_v3 import InceptionV3

from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions
from keras.applications.mobilenet_v2 import MobileNetV2

from keras.applications.xception import preprocess_input
from keras.applications.xception import decode_predictions
from keras.applications.xception import Xception

from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import VGG19

from keras.applications.densenet import preprocess_input
from keras.applications.densenet import decode_predictions
from keras.applications.densenet import DenseNet201

from keras.applications.densenet import preprocess_input
from keras.applications.densenet import decode_predictions
from keras.applications.densenet import DenseNet201

from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.inception_resnet_v2 import decode_predictions
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions
from keras.applications.mobilenet_v2 import MobileNetV2

from keras.applications.nasnet import preprocess_input

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

models = []
models = [VGG16(weights='imagenet', include_top=False)]
models_name = ['VGG16']

csv_ok = pd.read_csv('d:\\dane\\HARRISON\\data_list.txt', header=None)
for model_id in range(len(models)):
    base_model = models[model_id]
    # print(models_name[model_id])
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    model = Model(inputs=base_model.input, outputs=x)

    '''
    classes_list = []
    print(csv_ok.shape)
    for a in range(csv_ok.shape[0]):
        #print(str(csv_ok[0][a]))
        classes_list.append(str(csv_ok[0][a]).split('/')[1])
    classes_list = list(set(classes_list))
    print(classes_list)
    '''
    # print(csv_ok.shape)
    path_to_data = 'd:\\dane\\HARRISON\\'

    # for a in range(1):
    for a in range(csv_ok.shape[0]):
        if a % 100 == 0:
            print(models_name[model_id] + " " + str(a) + " of " + str(csv_ok.shape[0]))
        my_file = str(csv_ok.iloc[a, 0])
        img_path = path_to_data + my_file
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)

        file_object = open('../Features/' + models_name[model_id] + 'Features.txt', 'a')
        features = features[0]
        for b in range(features.shape[0]):
            if b > 0:
                file_object.write(",")
            file_object.write(str(features[b]))
        file_object.write('\n')
        file_object.close()
        # print(models_name[model_id] + " " + str(features.shape))
