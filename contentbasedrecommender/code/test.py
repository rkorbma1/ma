import pandas as pd
import time
import numpy as np
import json
from gensim.models import KeyedVectors
import heapq
import collections

dataframe = pd.read_csv('titatest.csv')
model = KeyedVectors.load('word2vec_german.model')
l = [json.loads(w) for w in dataframe.vectors]
m = [json.loads(w) for w in dataframe.preprocess]
n = [json.loads(w) for w in dataframe.nouns]

start = time.time()

#o = [json.loads(w) for w in dataframe.nouns_vectors]
#p = [json.loads(w) for w in dataframe.preprocess_vectors]
from sklearn.metrics import pairwise_distances

dist_out = 1-pairwise_distances(l, metric="cosine")

top_10_ind = dist_out[0].argsort()[-10:][::-1]

i=1
print('Query: ', dataframe.article_trans_teaser[0])
for value in range(1,len(top_10_ind)):
    print(value)
    print(str(i)+ 'th Similar Item: ', dataframe.article_trans_teaser[value])
    i=i+1
    x=3
    print('Die folgenden Worte sind die ähnlichsten in den beiden Artikeln.')
    sim_dict = {}
    for v in n[0]:
        for b in n[top_10_ind[value]]:
            words = str(v) + ' ' + str(b)
            sim_dict[words] = model.similarity(v, b)
    top_3_sim_dict = (
    {key: value for key, value in sim_dict.items() if value in heapq.nlargest(3, sim_dict.values())})
    sorted_dict = collections.OrderedDict(top_3_sim_dict)
    for k, v in sorted_dict.items():
        keys = k.split()
        print('Wort aus dem gelikten Beitrag: ' + str(
            keys[0]) + '. Das ähnliche Wort aus dem vorgeschlagenen Beitrag: ' + str(
            keys[1]) + '. Mit dem Ähnlichkeitswert: ' + str(v) + '.')

end = time.time()

print(end-start)

#dataframe.drop(columns=['nouns_vectors','preprocess_vectors'], inplace=True)
"""
k = [[0.1,0.2,0.3],[0.1,0.2,0.3]]

l = json.dumps(k)

print(l)
print(type(l))

li = json.loads(l)

print(li)
print(type(li))

for vals in li[0]:
    print(type(vals))

#code laufen lassen für die beiden vectorenzeilen. 

preprocess_vectors=[]

for vals in m:
    vectors=[]
    for v in vals:
        vector = model[v].tolist()
        vectors.append(vector)
    preprocess_vectors.append(json.dumps(vectors))

dataframe['preprocess_vectors']= preprocess_vectors

nouns_vectors=[]

for vals in n:
    vectors=[]
    for v in vals:
        vector = model[v].tolist()
        vectors.append(vector)
    nouns_vectors.append(json.dumps(vectors))

dataframe['nouns_vectors']= nouns_vectors

#dataframe.to_csv('titatest.csv')
"""
