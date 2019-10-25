import os
import pandas as pd
from sqlalchemy import create_engine
import json

base_dir = os.path.abspath(os.path.join(__file__, "../.."))
engine = create_engine('sqlite:///' + os.path.abspath(os.path.join(base_dir, 'data.sqlite')))
dataframe = pd.read_sql_table('items', con=engine)
from sklearn.feature_extraction.text import CountVectorizer

print(dataframe.columns)

real_authors =[]

for values in dataframe.authors:
    if ',' in values:
        authors= values.split(', ')
        #real_authors.append(json.dumps(authors))
        real_authors.append(authors)
    else:
        authors = [values]
        #real_authors.append(json.dumps(authors))
        real_authors.append(authors)

dataframe['authors'] = real_authors

count_vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)

XY = count_vectorizer.fit_transform(real_authors)
tfidf_array = XY.toarray()
df = pd.DataFrame(tfidf_array, columns=count_vectorizer.get_feature_names(), index=dataframe.audio_id)

print(df['Natalie Böneke'])

print(df.head())

#Einbauen, wie schon im Profile_Learner gemacht mit den tags!