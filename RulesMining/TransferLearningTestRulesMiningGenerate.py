#Order of execution: #8
import pandas as pd
df = pd.read_csv('RulesRandom0.0001.csv')
print(df.columns)


print("df.shape" + str(df.shape))
treshs = [0.2, 0.3, 0.5, 0.75, 0.95]

for tresh in treshs:
    df = df[df['confidence'] > tresh]
    print("df.shape" + str(df.shape))
    all_antecedents = []
    all_consequents = []
    all_confidence = []

    def frozenset_to_set(my_str):
        my_str = my_str.replace('frozenset({','')
        my_str = my_str.replace('})','')
        x = my_str.split(", ")
        for id_x in range(len(x)):
            x[id_x] = float(x[id_x])
        return x

    for a in range(df.shape[0]):
        if a % 100 == 0:
            print(str(a) + " " + str(df.shape[0]))
        all_antecedents.append(frozenset_to_set(df.iloc[a,1]))
        all_consequents.append(frozenset_to_set(df.iloc[a,2]))
        all_confidence.append(df.iloc[a,6])

    print(len(all_antecedents))
    print(len(all_consequents))

    import pickle
    with open('all_antecedentsRandom0.0001_' + str(tresh) + '.bin', 'wb') as fp:
        pickle.dump(all_antecedents, fp)
    with open('all_consequentsRandom0.0001_' + str(tresh) + '.bin', 'wb') as fp:
        pickle.dump(all_consequents, fp)
    with open('all_confidenceRandom0.0001_' + str(tresh) + '.bin', 'wb') as fp:
        pickle.dump(all_confidence, fp)


