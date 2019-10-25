from contentbasedrecommender import app, db
from sqlalchemy import create_engine, desc, asc
from contentbasedrecommender.code.ProfileLearner import ProfileLearner
from contentbasedrecommender.code.Recommender import Recommender
import numpy as np
from contentbasedrecommender.models import User,Item, Rating, Recommendations
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
import os
import json
import psutil
from datetime import datetime

datetime = datetime.utcnow
print(datetime)
print(type(datetime))
"""
base_dir = os.path.abspath(os.path.join(__file__, "../contentbasedrecommender"))
engine = create_engine('sqlite:///' + os.path.abspath(os.path.join(base_dir, 'data.sqlite')))

db.create_all()
all_entries = Rating.query.filter_by(user_id=2).order_by(desc(Rating.rating_date)).all()
last_eight_entries = all_entries[:8:]
l=[]
for values in last_eight_entries:
    print(values.item_id)
    print(values.rating)
    l.append(values.item_id)
print(l)
l.reverse()
print(l)
"""
"""
profile_learner = ProfileLearner()
recommender = Recommender()
df = pd.read_sql_table('items', engine, index_col='audio_id')
#df['final_vectors'] = [json.loads(w) for w in df.vector]
df['final_preprocess'] = [json.loads(w) for w in df.preprocess]
df['final_nouns'] = [json.loads(w) for w in df.nouns]

for values in df['final_nouns']:
    for value in values:
        if(len(value) <= 1):
            print(values)
            print(len(value))
            print(value)
        print(values)
"""
"""
authors =[]

for values in df['authors']:
    if(',' in values):
        authors.append(values.split(', '))
    else:
        authors.append([values])

df['list_authors'] = authors

def cosine_similarity(A):
    Anorm = A / np.linalg.norm(A, axis=-1)[:, np.newaxis]
    return linear_kernel(Anorm)
"""
"""
vectors = df['final_vectors'].tolist()
sim_matrix = cosine_similarity(np.asarray(vectors))
#print(sim_matrix.nbytes)
np.save('sim_matrix.npy', sim_matrix)
"""
"""
sim_matrix = np.load('sim_matrix.npy')
print(type(sim_matrix))
print(sim_matrix.nbytes)

process = psutil.Process(os.getpid())
print(process.memory_info().rss)

print(process.memory_info()[0])
"""