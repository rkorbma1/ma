import pandas as pd
import os
from sqlalchemy import create_engine
from datetime import datetime
from contentbasedrecommender.models import Item
from contentbasedrecommender import db
from contentbasedrecommender.code.ContentGetter import ContentGetter

class ContentAnalyzer:
    #Get the dataframe from the CMS-API.
    def __init__(self, engine, first_setup=True):
        self.first_setup = first_setup
        self.engine = engine

    def import_dataframe(self):
        if(self.first_setup==True):
            dataframe = pd.read_sql_table('API_old', con= self.engine, columns=['audio_authors','audio_duration','audio_path_abs','audio_time','audio_time_iso','broadcast_id','date','article_id','article_trans_teaser','audio_title'], index_col='article_id')
        else:
            try:
                old_dataframe = pd.read_sql_table('API_old', con=self.engine,
                                              columns=['audio_authors','audio_title', 'audio_duration', 'audio_path_abs',
                                                       'audio_time', 'audio_time_iso', 'broadcast_id', 'date',
                                                       'article_id', 'article_trans_teaser'])
                dataframe = pd.read_sql_table('API_new', con=self.engine,
                                              columns=['audio_authors', 'audio_duration', 'audio_path_abs',
                                                       'audio_time', 'audio_time_iso', 'broadcast_id', 'date',
                                                       'article_id', 'article_trans_teaser','audio_title'])
                self.final_dataframe= pd.concat([old_dataframe, dataframe])
                print(self.final_dataframe.shape)
            except:
                print('Es ist ein Fehler aufgetreten!')
        zeilen = dataframe.shape[0]
        dataframe.dropna(subset=['article_id'], inplace=True)
        dataframe.dropna(subset=['article_trans_teaser'], inplace=True)
        dataframe.dropna(subset=['audio_authors'], inplace=True)
        if(self.first_setup==True):
            print('Der Dataframe mit dem Pfad API_old hat', dataframe.shape[0] , 'Zeilen und', dataframe.shape[1], 'Spalten.')
            print('Das heißt, dass genau', round((1-(dataframe.shape[0]/zeilen))*100,2), '% des eigentlichen Dataframes gelöscht wurden, da die inhaltlichen Informationen der gedroppten Audios für einen Content-Based Recommender nicht ausreichen.')
        else:
            print('Der alte Dataframe mit dem Pfad API_old hat', old_dataframe.shape[0], 'Zeilen und',
                  old_dataframe.shape[1], 'Spalten.' )
            print('Der Dataframe mit dem Pfad API_new hat', dataframe.shape[0], 'Zeilen und',
                  dataframe.shape[1], 'Spalten.')
            print('Das heißt, dass genau', round((1 - (dataframe.shape[0] / zeilen)) * 100, 2),
                  '% des eigentlichen Dataframes gelöscht wurden, da die inhaltlichen Informationen der gedroppten Audios für einen Content-Based Recommender nicht ausreichen.')
        dataframe.set_index(['article_id'],inplace=True)
        dataframe.index = dataframe.index.astype(int)
        dataframe_authors = self.get_right_authors(dataframe)
        print(dataframe_authors.head())
        dataframe_authors.to_csv('right_authors.csv')


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
        dataframe['audio_authors'] = new_author
        print(dataframe.shape)
        #result = self.get_keywords(dataframe)
        #return result
        self.insert_into_items(dataframe)

#TODO: Keyword Stemming: Von gehst zu geh...
    def get_keywords(self, dataframe):
        import requests
        import itertools
        #We get the keywords with the Dandelion-API --> For further information look at this link: https://dandelion.eu/
        DANDELION_TOKEN = '9d047a49b15a419eb05403b69b639744'
        ENTITY_URL = 'https://api.dandelion.eu/datatxt/nex/v1'
        def get_entities(text, confidence=0.2, lang='de'):
            payload = {
                'token': DANDELION_TOKEN,
                'text': text,
                'min_confidence': confidence,
                'lang': lang,
                'social.hashtag': True,
                'social.mention': True,
                'min_length ': 1
            }
            response = requests.get(ENTITY_URL, params=payload)
            return response.json()

        #We got a limit of 1000 dandelion API-Tokens per day!
        limit = 1000
        dataframe = dataframe
        #dataframe[:1000]
        #https://stackoverflow.com/questions/16396903/delete-the-first-three-rows-of-a-dataframe-in-pandas
        dataframe['keywords'] =''
        for index, row in itertools.islice(dataframe.iterrows(), limit):
            query = row['article_trans_teaser']
            keywords_list = []
            response = get_entities(query)
            lists_of_dicts = response['annotations']
            for ind in range(len(lists_of_dicts)):
                dict = lists_of_dicts[ind]
                keywords_list.append(dict['spot'])
                keywords_list = [x.replace(" ", "") for x in keywords_list]
            keyword = ' '.join(keywords_list)
            dataframe.at[index, 'keywords'] = keyword
        dataframe.to_csv('keywords_test.csv')
        self.insert_into_items(dataframe)
        return dataframe

    def insert_into_items(self, dataframe):
        keywords = dataframe

        print(keywords.columns)

        for values in keywords['audio_authors']:
            print(values)

        keywords.dropna(inplace=True, subset=['keywords'])
        keywords.dropna(inplace=True, subset=['audio_authors'])
        keywords.dropna(inplace=True, subset=['audio_title'])
        keywords.dropna(inplace=True, subset=['audio_authors'])

        print(keywords.shape)

        """
        for index, row in keywords.iterrows():
            timestamp = int(row['audio_time'])
            date = datetime.fromtimestamp(timestamp)
            print(type(date))
            real_datetime.append(date.to_datetime())

        keywords['audio_time_real'] = real_datetime
        """

        keywords['audio_time_real'] = pd.to_datetime(
            keywords['audio_time'].apply(lambda x: datetime.fromtimestamp(int(x))))

        print(keywords.head())

        print(keywords.columns)

        #unique_authors = list(keywords['audio_authors'].unique())

        #print(unique_authors)
        """
        final_unique_authors = []

        for values in unique_authors:
            if (',' in values):
                abc = values.split(',')
                print(abc)
                for v in abc:
                    final_unique_authors.append(v)
            else:
                print(values)
                final_unique_authors.append(values)
        print(final_unique_authors)
        for names in final_unique_authors:
            result = Author.query.filter_by(name=names).first()
            print(names)
            print(result)
            if (result is None):
                entry = Author(names)
                db.session.add(entry)
                db.session.commit()
            else:
                print('Gibts schon!')
        """
        for index, row in keywords.iterrows():
            result = Item.query.filter_by(audio_id=index).first()
            print(index)
            print(row['audio_title'])
            print(row['audio_time_real'])
            print(type(row['audio_time_real']))
            if result is None:
                entry = Item(audio_id=index, title=row['audio_title'], url=row['audio_path_abs'],
                             teaser=row['article_trans_teaser'], broadcast_date=row['audio_time_real'],
                             duration=row['audio_duration'], keywords=row['keywords'], authors= row['audio_authors'])
                db.session.add(entry)
                db.session.commit()
        """
        for index, row in keywords.iterrows():
            authors = row['audio_authors']
            id = index
            print(id)
            entry = Item.query.get(id)
            print(authors)
            if (',' in authors):
                abc = authors.split(',')
                for v in abc:
                    author = Author.query.filter_by(name= v).first()
                    entry.authors.append(author)
            else:
                authors = authors.replace("'", '')
                author = Author.query.filter_by(name=authors).first()
                entry.authors.append(author)
            db.session.commit()
            #self.final_dataframe.to_sql('API_old', con=self.engine, if_exists='replace')
        """
"""
def import_new_data():
    url_broadcasts='https://srv.deutschlandradio.de/i.2745.de.rpc'
    url_audios= 'https://srv.deutschlandradio.de/i.2744.de.rpc?drau:broadcast_id='
    base_dir = os.path.abspath(os.path.join(__file__, "../.."))
    engine = create_engine('sqlite:///' + os.path.abspath(os.path.join(base_dir, 'data')))
    contentgetter = ContentGetter(url_broadcasts=url_broadcasts, url_audios= url_audios ,first_setup=True, engine= engine)
    contentgetter.compare()
    contentanalyzer = ContentAnalyzer(first_setup=False, engine= engine)
    contentanalyzer.import_dataframe()
"""
#import_new_data()
db.create_all()

base_dir = os.path.abspath(os.path.join(__file__, "../.."))
engine = create_engine('sqlite:///' + os.path.abspath(os.path.join(base_dir, 'data')))
dataframe = pd.read_sql_table('API_old', con=engine)
second_dataframe = pd.read_csv('keywords_dandelion.csv', usecols=['keywords','article_id'])
print(second_dataframe.head())
contentanalyzer = ContentAnalyzer(first_setup=False, engine=engine)
dataframe['article_id'] = pd.to_numeric(dataframe['article_id'])
dataframe = pd.merge(dataframe, second_dataframe, left_on='article_id', right_on='article_id', how='right')
dataframe.set_index('article_id',inplace=True)
print(dataframe.shape)
print(dataframe.head())
print(dataframe.columns)
contentanalyzer.get_right_authors(dataframe)

"""
scheduler = BlockingScheduler()
scheduler.add_job(import_new_data, 'interval', hours=24)
scheduler.start()
"""