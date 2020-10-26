#Order of execution: #11
import numpy as np
result = np.load('../Results/resultfrom6.bin.npy')
import numpy as np
actual = np.load('../Features/HashtagsLabels.bin.npy')
print(result.shape)
print(actual.shape)

import random
random.seed(4321)
my_random_sample = random.sample(range(actual.shape[0]), 5000)
actual = actual[my_random_sample,:]

all_results = np.zeros([5000, 7])

for a in range(actual.shape[0]):
    if a % 100 == 0:
        print(str(a) + " of " + str(actual.shape[0]))

    my_y = actual[a]
    my_y = set(np.nonzero(my_y)[0])
    #print(my_y)

    my_res = result[a, :]

    for aa in range(7):
        my_res_list = []
        for b in range(5 *aa, (5 * (aa + 1))):
            my_res_list.append(my_res[b])
        my_res_set = set(my_res_list)
        all_results[a, aa] = len(my_y.intersection(my_res_set))
np.savetxt('data_to_graph.csv', all_results, delimiter=',',fmt='%1.0f')
