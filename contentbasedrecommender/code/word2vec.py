import os
from sqlalchemy import create_engine,desc
import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
import time
from contentbasedrecommender.models import User,Item, Rating, Recommendations

base_dir = os.path.abspath(os.path.join(__file__, "../../"))
engine = create_engine('sqlite:///' + os.path.abspath(os.path.join(base_dir, 'data.sqlite')))
df = pd.read_sql_table('items', engine, index_col='audio_id')

all_entries = Rating.query.order_by(desc(Rating.rating_date)).filter_by(user_id=3).all()
last_eight_entries = all_entries[:8:]
item_id_list =[]
rating_list=[]
for values in last_eight_entries:
    item_id_list.append(values.item_id)
    rating_list.append(values.rating)
item_title_list = list(df.loc[item_id_list,:].title)

print(item_id_list)
print(item_title_list)
"""
df = pd.read_sql_table('items', engine, index_col='audio_id')
df['final_vectors'] = [json.loads(w) for w in df.vector]
df['final_preprocess'] = [json.loads(w) for w in df.preprocess]
df['final_nouns'] = [json.loads(w) for w in df.nouns]
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

start_1 = time.time()
vectors = df['final_vectors'].tolist()
sim_matrix = cosine_similarity(np.asarray(vectors))
end_1 = time.time()
print(end_1 -start_1)

#sim_matrix_df = pd.DataFrame(data= sim_matrix, columns=df.index, index=df.index)
start = time.time()
ratings_df = pd.read_sql_table('ratings', engine, index_col='rating_id')
ratings_df = ratings_df[ratings_df.user_id == 3]
liked_ratings_indices = ratings_df[ratings_df.rating == 1]
if (len(liked_ratings_indices) >= 1):
    liked_items = df[df.index.isin(liked_ratings_indices.item_id)].index
    not_seen_items = df[~df.index.isin(ratings_df.item_id)].index
    unseen_items = df[~df.index.isin(ratings_df.item_id)]
    print(not_seen_items.shape)
    indices = list(df.index)
    liked_iloc= np.asarray([indices.index(v) for v in liked_items])
    not_seen_iloc = np.asarray([indices.index(v) for v in not_seen_items])
    result = sim_matrix[liked_iloc.ravel()]
    max_sim = np.amax(np.asarray(result), axis=0)
    max_not_seen = max_sim[not_seen_iloc]
    max_not_seen_ind = np.argpartition(max_not_seen, -30)[-30:]
    array_indices = np.asarray(not_seen_items)
    final_indices = list(not_seen_items[max_not_seen_ind].values)
"""
"""
    print(type(final_indices.values))
"""
"""   
    final_recommendations = unseen_items.loc[unseen_items.index.isin(final_indices)]
    print(final_recommendations)
    #final_recommendations = [unseen_items[i.item()] for i in final_indices]

end = time.time()
print(end -start) # ~4.701 seconds
"""