import pandas as pd
from sqlalchemy import create_engine
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import time

class ProfileLearner:

    #Get the dataframe from the CMS-API.
    def __init__(self):
        base_dir = os.path.abspath(os.path.join(__file__, "../.."))
        self.engine = create_engine('sqlite:///'+ os.path.abspath(os.path.join(base_dir,'data')))

    def get_user_preferences(self, items_df, liked_ratings_indices):
        #Brauche ich nicht!
        corpus = items_df['final_nouns']
        vectorizer = TfidfVectorizer(use_idf=True, tokenizer=lambda i:i, lowercase=False)
        XY = vectorizer.fit_transform(corpus)
        tfidf_array = XY.toarray()
        df = pd.DataFrame(tfidf_array, columns=vectorizer.get_feature_names(), index=items_df.index)
        #Ab hier zählt es!
        #Time --> 0.2579941749572754
        start = time.time()
        final_df = df.loc[liked_ratings_indices.item_id,:]
        f = final_df.sum(axis=0, skipna=True).to_dict()
        what_user_likes_keys = [k for k, v in f.items() if v > 0]
        what_user_likes_values = [v for k, v in f.items() if v > 0]
        final_list=[]
        for i in range(0,len(what_user_likes_keys)):
            final_dict ={}
            final_dict['x'] = what_user_likes_keys[i]
            final_dict['value'] = what_user_likes_values[i]
            final_list.append(final_dict)
        end = time.time()
        print(end - start)
        return final_list

    def get_user_author_preferences(self, items_df, liked_ratings_indices):
        #Brauche ich nicht!
        real_authors = items_df.list_authors
        count_vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
        XY = count_vectorizer.fit_transform(real_authors)
        count_array = XY.toarray()
        df = pd.DataFrame(count_array, columns=count_vectorizer.get_feature_names(), index=items_df.index)#Ab hier zählt es!
        #Time --> 0.2579941749572754
        start = time.time()
        final_df = df.loc[liked_ratings_indices.item_id,:]
        f = final_df.sum(axis=0, skipna=True).to_dict()
        authors_user_likes_keys = [k for k, v in f.items() if v > 0]
        authors_user_likes_values = [v for k, v in f.items() if v > 0]
        final_author_list=[]
        for i in range(0,len(authors_user_likes_keys)):
            final_dict ={}
            final_dict['x'] = authors_user_likes_keys[i]
            final_dict['value'] = authors_user_likes_values[i]
            final_author_list.append(final_dict)
        end = time.time()
        print(final_author_list)
        return final_author_list

"""
    def get_right_authors(self,dataframe):
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

        #Hier wird ein eine Dataframe-Zeile mit den 'richtigen'-AutorenNamen erstellt, da diese im Dataframe teilweise unterschiedlich gespeichert sind. Beispiel: Wolfgang Rathert / Olaf Wilhelmer für ein Audio und zum Beispiel Michael Stegemann, Olaf Wilhelmer in einem anderen.

        dataframe['cleaned_audio_authors'] = ''
        dataframe.dropna(subset=['authors'], inplace=True)
        print(dataframe.shape)
        for index, row in dataframe.iterrows():
            author = row['authors']
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
        dataframe.drop(['authors', 'cleaned_audio_authors'], axis=1, inplace=True)
        dataframe['authors'] = new_author
        print(dataframe.shape)
        #result = self.get_keywords(dataframe)
        #return result
        return dataframe
"""
    #TODO: Do the same for authors! Is really, really fast!
    #TODO: Make it faster by creating TFIDF-Matrix in app.py!
    #TODO: Vllt nur nouns nehmen für die WordCloud!

#profile = ProfileLearner()
#profile.get_user_preferences()

