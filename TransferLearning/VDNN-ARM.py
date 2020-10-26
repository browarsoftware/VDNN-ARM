#Order of execution: #9
from itertools import chain

#df = pd.read_csv('../RulesMining/RulesRandom0.0001.csv')

all_antecedents = []
all_consequents = []

import pickle

res_files = ['results_02_bin', 'results_3_bin', 'results_5_bin', 'results_75_bin', 'results_95_bin']
procs = [0.2, 0.3, 0.5, 0.75, 0.95]
proc = 0
for id_help in range(len(procs)):
    proc = procs[id_help]
    file_to_save_res = res_files[id_help]

    with open ('../RulesMining/all_antecedentsRandom0.0001_' + str(proc) + '.bin', 'rb') as fp:
        all_antecedents = pickle.load(fp)

    with open ('../RulesMining/all_consequentsRandom0.0001_' + str(proc) + '.bin', 'rb') as fp:
        all_consequents = pickle.load(fp)

    with open ('../RulesMining/all_confidenceRandom0.0001_' + str(proc) + '.bin', 'rb') as fp:
        all_confidence = pickle.load(fp)

    def filter_best(how_much, my_res):
        (values,counts) = np.unique(my_res,return_counts=True)

        my_res = my_res.tolist()
        result = sorted(my_res, key=my_res.count,
                        reverse=True)
        (values, counts) = np.unique(result, return_counts=True)

        f = sorted(range(len(counts)), key=lambda k: counts[k],
                   reverse=True)

        bbb = values[f]
        ccc = counts[f]

        values_list = []
        for a in range(len(bbb)):
            if ccc[a] > 1:
                values_list.append(bbb[a])

        return_list = []
        loop_1 = min(how_much, len(values_list))
        for a in range(loop_1):
            return_list.append(values_list[a])

        loop_2 = how_much - loop_1
        if loop_2 < 1:
            loop_2 = 0
        return [np.asarray(return_list), loop_2]

    import numpy as np
    actual = np.load('../Features/HashtagsLabels.bin.npy')
    import random
    random.seed(4321)
    my_random_sample = random.sample(range(actual.shape[0]), 5000)

    actual = actual[my_random_sample,:]

    import numpy as np
    result = np.load('../Results/resultfrom6.bin.npy')

    actual_length = []
    predicted_length = []

    #bool_limit_output = False
    #bool_limit_input = False

    for how_much in [5]:
        all_my_pred = []
        for my_id in range(actual.shape[0]):
            my_y = actual[my_id]
            my_y = np.nonzero(my_y)[0]
            my_res = result[my_id,:]
            ll = []

            (ll, loop_2) = filter_best(len(my_res), my_res)

            ll_add = []
            ll_set = set(ll)
            ll_confidence = []
            for b in range(len(all_antecedents)):
                bb = set(all_antecedents[b])
                if bb.issubset(ll_set):
                    ll_add.append(all_consequents[b])
                    ll_confidence.append(all_confidence[b])
            ll_add = list(chain(*ll_add))

            Z = [x for _, x in sorted(zip(ll_confidence, ll_add), reverse=True)]

            ll_set = set(ll)
            id_add = 0

            while id_add < len(Z):
                ll_set.add(Z[id_add])
                id_add = id_add + 1


            ll = list(ll_set)

            print(str(my_id) + " " + str(len(ll)))
            all_my_pred.append(ll)
            print(ll)

    import pickle
    with open(file_to_save_res, "wb") as fp:   #Pickling
        pickle.dump(all_my_pred, fp)


