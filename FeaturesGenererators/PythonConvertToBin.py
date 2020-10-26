#Order of execution: #4
import numpy as np
models_name = ['NASNetLarge', 'InceptionV3', 'MobileNetV2', 'Xception', 'DenseNet201', 'InceptionResNetV2', 'VGG16_Places365']#, 'VGG19', 'InceptionResNetV2', 'NASNetLarge']
from numpy import genfromtxt
model_id = 0

print('Loadind data ' + models_name[model_id])

for model_id in range(0, len(models_name)):
    my_data = genfromtxt('../Features/'+ models_name[model_id] +'Features.txt', delimiter=',', skip_header=0)
    print(my_data.shape)
    np.save('../FeaturesBin/' + models_name[model_id] + '.bin', my_data)
#np.savetxt('Features/all_features_from_DNN.txt', my_data, delimiter=',')
#np.save('Features/all_features_from_DNN.bin', my_data)
