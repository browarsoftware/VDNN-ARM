#Order of execution: #13
import numpy as np
models_name = ['VGG16', 'VGG16_Places365']
from numpy import genfromtxt
model_id = 0

print('Loadind data ' + models_name[model_id])
my_data = genfromtxt('../Features/'+ models_name[model_id] +'Features.txt', delimiter=',', skip_header=0)

print(my_data.shape)
for model_id in range(1, len(models_name)):
    print('Loadind data ' + models_name[model_id])
    my_data2 = genfromtxt('../Features/'+ models_name[model_id] +'Features.txt', delimiter=',', skip_header=0)
    my_data = np.c_[my_data, my_data2]
    print(my_data.shape)
#np.savetxt('Features/all_features_from_DNN.txt', my_data, delimiter=',')
np.save('../FeaturesBin/all_features_baseline_DNN.bin', my_data)