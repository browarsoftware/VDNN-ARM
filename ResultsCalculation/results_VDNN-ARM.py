#Order of execution: #10
def evaluate_arrays(a1, a2):

    K = set(np.asarray(a1))
    GT = set(np.asarray(a2))

    set_intersection = len(list(K.intersection(GT)))
    denominator = len(list(K))
    if denominator > 0:
        precision = set_intersection / len(list(K))
    else:
        precision = 0
    # print(precision)
    denominator = len(list(GT))
    if denominator > 0:
        recall = set_intersection / len(list(GT))
    else:
        recall = 0
    # print(recall)

    accuracy = 0
    if set_intersection > 0:
        accuracy = 1
    # print(accuracy)
    return [precision, recall, accuracy]




import numpy as np
actual = np.load('../Features/HashtagsLabels.bin.npy')
import random
random.seed(4321)
my_random_sample = random.sample(range(actual.shape[0]), 5000)

actual = actual[my_random_sample,:]

res_files = ['results_02_bin', 'results_3_bin', 'results_5_bin', 'results_75_bin', 'results_95_bin']

import pickle

class myFloat( float ):
    def __str__(self):
        return "%.2f"%self

file_to_save_res = 'evaluation_all_hastags.txt'
for file_id in range(len(res_files)):
    with open('../TransferLearning/' + res_files[file_id], "rb") as fp:  # Unpickling
        print(res_files[file_id])
        result = pickle.load(fp)
    for how_much_res in [1,2,3,4,5,100]:
        precision5 = 0
        recall5 = 0
        accuracy5 = 0
        print(how_much_res)
        for my_id in range(actual.shape[0]):
            my_res = np.asarray(result[my_id])
            my_res = my_res[0:how_much_res]

            my_y = actual[my_id]
            my_y = np.nonzero(my_y)[0]
            evaluate_res = evaluate_arrays(my_res, my_y)

            precision5 = precision5 + evaluate_res[0]
            recall5 = recall5 + evaluate_res[1]
            accuracy5 = accuracy5 + evaluate_res[2]


        precision5 = precision5 / actual.shape[0]
        recall5 = recall5 / actual.shape[0]
        accuracy5 = accuracy5 / actual.shape[0]
        f1 = 2 * precision5 * recall5 / (precision5 + recall5)

        precision5 = str(myFloat(precision5 * 100))
        recall5 = str(myFloat(recall5 * 100))
        accuracy5 = str(myFloat(accuracy5 * 100))
        f1 = str(myFloat(f1 * 100))

        file_object = open(file_to_save_res, 'a')
        file_object.write('No more than ' + str(how_much_res) + "&" + precision5 + "&" + recall5 + "&" + accuracy5 + "&" + f1 + "\\\\" + '\n')
