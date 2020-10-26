#Order of execution: #7
# http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules

import numpy as np
actual = np.load('../Features/HashtagsLabels.bin.npy')
print(actual.shape)


import random
random.seed(4321)
my_random_sample = random.sample(range(actual.shape[0]), 5000)
mask = np.ones(actual.shape[0], dtype=bool)
mask[my_random_sample] = False
actual = actual[mask, ]

print(actual.shape)

all_data = []
for my_id in range(actual.shape[0]):
    my_y = actual[my_id]
    my_y = list(np.nonzero(my_y)[0])
    all_data.append(my_y)

dataset = all_data

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.0001, use_colnames=True, low_memory=True)
ar = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.001)
ar.to_csv('RulesRandom0.0001.csv')
