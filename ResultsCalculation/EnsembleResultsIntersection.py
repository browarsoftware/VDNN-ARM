#Order of execution: #18

import numpy as np
#actual = np.load('../Hashtags50/Features/HashtagsLabels.bin.npy')
#print(actual[0,0])

import pandas as pd
from itertools import chain


import pickle
file_to_save_res = 'ensemble_results_intersection.txt'
#file_to_save_res = 'ensemble_results_union.txt'

#print(len(all_antecedents))
#print(len(all_consequents))

class myFloat( float ):
    def __str__(self):
        return "%.2f"%self

def most_frequent(List):
    return max(set(List), key = List.count)

def filter_best(how_much, my_res):
    (values,counts) = np.unique(my_res,return_counts=True)
    #print(values)
    #print(counts)

    values_list = []
    counts_list = []
    for a in range(len(counts)):
        if counts[a] > 1:
            values_list.append(values[a])
            counts_list.append(counts[a])

    #print(values_list)
    #print(counts_list)

    return_list = []
    loop_1 = min(how_much, len(values_list))
    for a in range(loop_1):
        return_list.append(values_list[a])

    loop_2 = how_much - loop_1
    if loop_2 < 1:
        loop_2 = 0
    return [np.asarray(return_list), loop_2]

def evaluate_arrays_intersection(a1, a2):

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

def evaluate_arrays_union(a1, a2):

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


import numpy as np
actual = np.load('../Features/HashtagsLabels.bin.npy')
import random
random.seed(4321)
my_random_sample = random.sample(range(actual.shape[0]), 5000)

actual = actual[my_random_sample,:]

import numpy as np
result = np.load('../Results/ensemble_results.bin.npy')

precision5 = 0
recall5 = 0
accuracy5 = 0

my_id = 0



#do_5 = True
how_much = 5
bool_append_minus_one = False
#for how_much in range(1,6):
#for how_much in range(5,6):
for how_much in range(1,6):

    precision5 = 0
    recall5 = 0
    accuracy5 = 0

    for my_id in range(actual.shape[0]):
    #for my_id in range(10):
        print(my_id)
        my_y = actual[my_id]
        my_y = np.nonzero(my_y)[0]
        my_res = result[my_id,:]
        ll = []
        #evaluate_res = evaluate_arrays(my_res, my_y)
        '''
        if how_much > 0:
                my_res_list = []
                for b in range(0,0+how_much):
                    my_res_list.append(my_res[b])
                for b in range(5,5+how_much):
                    my_res_list.append(my_res[b])
                for b in range(10,10+how_much):
                    my_res_list.append(my_res[b])
                my_res = np.asarray(my_res_list)
        '''
        aa = 0
        
        inter_set = set(my_res[aa:(aa+5)])
        ll = int((len(my_res) / 5) - 1)
        for a in range(1,ll):
            aa = 5 * a
            vv = my_res[aa:(aa + 5)]
            inter_set.intersection(vv)

        my_res = np.asarray(list(inter_set))[0:how_much]
        #my_res = np.asarray(list(set(my_res)))[0:how_much]
        print(len(my_res))
        my_res = my_res
        evaluate_res = evaluate_arrays_intersection(my_res, my_y)
        #evaluate_res = evaluate_arrays_union(my_res, my_y)
        print(evaluate_res)
        #print(evaluate_res)
        precision5 = precision5 + evaluate_res[0]
        recall5 = recall5 + evaluate_res[1]
        accuracy5 = accuracy5 + evaluate_res[2]




    file_object = open(file_to_save_res, 'a')
    #file_object.write('how_much=' + str(how_much) + "\n")
    precision5 = precision5 / actual.shape[0]
    recall5 = recall5 / actual.shape[0]
    accuracy5 = accuracy5 / actual.shape[0]
    f1 = 2 * precision5 * recall5 / (precision5 + recall5)

    precision5 = str(myFloat(precision5 * 100))
    recall5 = str(myFloat(recall5 * 100))
    accuracy5 = str(myFloat(accuracy5 * 100))
    f1 = str(myFloat(f1 * 100))

    file_object.write('No more than ' + str(how_much) + "&" + precision5 + "&" + recall5 + "&" + accuracy5 + "&" + f1 + "\\\\" + '\n')
    #file_object.write('precision5='+str(precision5) + "\n")
    #file_object.write('recall5='+str(recall5) + "\n")
    #file_object.write('accuracy5='+str(accuracy5) + "\n")
    #file_object.write('f1=' + str(accuracy5) + "\n")
    file_object.close()
