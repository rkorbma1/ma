from contentbasedrecommender import db
from models import User, Item, Rating, Vectors
import pandas as pd
from datetime import datetime
import os
from sqlalchemy import create_engine
import json
from gensim.models import KeyedVectors

def get_right_authors(dataframe):
    from nameparser import HumanName
    import re, math
    from collections import Counter

    WORD = re.compile(r'\w+')

    def text_to_vector(text):
        return Counter(WORD.findall(text))

    def get_similarity(a, b):
        a = text_to_vector(a.strip().lower())
        b = text_to_vector(b.strip().lower())
        return get_cosine(a, b)

    def get_cosine(vec1, vec2):
        # print vec1, vec2
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    # Hier wird ein eine Dataframe-Zeile mit den 'richtigen'-AutorenNamen erstellt, da diese im Dataframe teilweise unterschiedlich gespeichert sind. Beispiel: Wolfgang Rathert / Olaf Wilhelmer für ein Audio und zum Beispiel Michael Stegemann, Olaf Wilhelmer in einem anderen.

    dataframe['cleaned_audio_authors'] = ''
    dataframe.dropna(subset=['audio_authors'], inplace=True)
    print(dataframe.shape)
    for index, row in dataframe.iterrows():
        author = row['audio_authors']
        splitting_delimeters = r"[;/]"
        print(author)
        print(type(author))
        split = re.split(splitting_delimeters, author)
        if (len(split) == 1):
            name = HumanName(split[0])
            if (str(name.first) != ''):
                first_name = str(name.first)
                first_name = re.sub(r'[^\w\s]', '', first_name)
            if (str(name.last != '')):
                last_name = str(name.last)
                last_name = re.sub(r'[^\w\s]', '', last_name)
            dataframe.at[index, 'cleaned_audio_authors'] = (
                    str(name.title) + ' ' + first_name + ' ' + last_name + ' ' + str(name.suffix) + ' ' + str(
                name.nickname)).rstrip().lstrip()
        else:
            audio_authors = []
            for vals in split:
                name = HumanName(vals)
                if (str(name.first) != ''):
                    first_name = str(name.first)
                    first_name = re.sub(r'[^\w\s]', '', first_name)
                if (str(name.last != '')):
                    last_name = str(name.last)
                    last_name = re.sub(r'[^\w\s]', '', last_name)
                audio_authors.append(str(
                    name.title + ' ' + first_name + ' ' + last_name + ' ' + name.suffix + ' ' + name.nickname).rstrip().lstrip())
                dataframe.at[index, 'cleaned_audio_authors'] = ', '.join(audio_authors)

    # Replaces %90 and more similar strings
    def replace_similar_authors(input_list):
        for count, item in enumerate(input_list):
            rest_of_input_list = input_list[:count] + input_list[count + 1:]
            new_list = []
            for other_item in rest_of_input_list:
                similarity = get_similarity(item, other_item)
                if similarity >= 0.7:
                    new_list.append(item)
                else:
                    new_list.append(other_item)
            input_list = new_list[:count] + [item] + new_list[count:]

        return input_list

    new_author = replace_similar_authors(list(dataframe['cleaned_audio_authors']))
    dataframe.drop(['audio_authors', 'cleaned_audio_authors'], axis=1, inplace=True)
    dataframe['audio_authors'] = new_author
    print(dataframe.shape)
    # result = self.get_keywords(dataframe)
    # return result
    return dataframe

base_dir = os.path.abspath(os.path.join(__file__))
engine = create_engine('sqlite:///' + os.path.abspath(os.path.join('data.sqlite')))

db.create_all()
df = pd.read_csv('./code/titatest.csv')

nouns = [json.loads(w) for w in df.nouns]

flat_nouns = [x for sublist in nouns for x in sublist]

print(flat_nouns)

print(len(flat_nouns))

set_nouns = set(flat_nouns)

print(set_nouns)

print(len(set_nouns))

model = KeyedVectors.load('word2vec_german.model')

print("Fertig!")

for word in set_nouns:
  vector = model[word]
  print('Wort: ', word)
  print('Vektor: ', vector)
  vector_list = vector.tolist()
  string_vector = json.dumps(vector_list)
  entry = Vectors(word=word, vector=string_vector)
  db.session.add(entry)
  db.session.commit()

print(df.head())

df = get_right_authors(df)

print(df.columns)

#df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Unnamed: 0.1.1.1',
       #'Unnamed: 0.1.1.1.1'], inplace=True)
database_df = df[['audio_id','audio_title','audio_path_abs','image_small','image_large','vectors','preprocess','nouns','article_trans_teaser','date','audio_duration','audio_authors']]

print(len(database_df.audio_id))

database_df.drop_duplicates(subset=['audio_id'], inplace=True)

print(database_df.columns)

for index, row in database_df.iterrows():
    audio_id = row['audio_id']
    title= row['audio_title']
    url = row['audio_path_abs']
    small_picture_url = row['image_small']
    big_picture_url = row['image_large']
    vectors = row['vectors']
    preprocess = row['preprocess']
    nouns = row['nouns']
    teaser = row['article_trans_teaser']
    duration = float(row['audio_duration'])
    broadcast_date = row['date']
    broadcast_date = datetime.strptime(broadcast_date, '%Y-%m-%d').date()
    print(broadcast_date)
    print(type(broadcast_date))
    authors = row['audio_authors']
    entry = Item(audio_id,title, url, small_picture_url, big_picture_url, vectors, preprocess, nouns, teaser, duration, broadcast_date, authors)
    db.session.add(entry)
    db.session.commit()

#database_df.to_sql('items',con=engine,if_exists='append', index_label='audio_id')

