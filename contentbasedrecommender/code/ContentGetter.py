import requests
import pandas as pd
from pandas.io.json import json_normalize
import json
import numpy as np
from ast import literal_eval
import os
from sqlalchemy import create_engine

class ContentGetter:
    # Get the dataframe from the CMS-API.
    def __init__(self, url_broadcasts, url_audios, engine, first_setup=False):
        self.url_broadcasts = url_broadcasts
        self.url_audios = url_audios
        self.first_setup = first_setup
        self.engine = engine
"""
    def compare(self):
        old_dataframe = pd.read_sql_table('API_allbroadcasts_old', self.engine)
        print(old_dataframe.head())
        old_dataframe_all_ids =old_dataframe['broadcast_id'].unique().tolist()
        r = requests.get(self.url_broadcasts)
        json_data = r.json()
        all_broadcasts_df = pd.DataFrame.from_dict(json_data, orient='columns')
        all_broadcasts_df['broadcast_id'] = pd.to_numeric(all_broadcasts_df['broadcast_id'])
        all_broadcast_ids = all_broadcasts_df['broadcast_id'].unique().tolist()

        different_elements_1 = list(set(old_dataframe_all_ids) - set(all_broadcast_ids))
        different_elements_2 = list(set(all_broadcast_ids) - set(old_dataframe_all_ids))
        if not different_elements_1 and not different_elements_2:
            print('Die Listen sind die selben!')
            self.check_for_newer_audios(all_broadcast_ids)
        else:
            #TODO: Wenn die Listen nicht die Selben sind, dann hole dir alle Audios, die du kriegen kannst durch die API!
            if(len(different_elements_1) > 0):
                print('Dies sind die unterschiedlichen Elemente: ')
                print(different_elements_1)
            else:
                if (len(different_elements_2) > 0):
                    print('Dies sind die unterschiedlichen Elemente: ')
                    print(different_elements_2)

    def change_dataframe(self,dataframe):
        df = dataframe.drop('article', 1).join(dataframe.article.apply(pd.Series))
        del df[0]
        print(df.columns)
        df.to_sql('API_new', con=self.engine, if_exists='replace')
        #dataframe.to_csv('titatest.csv')

    def check_for_newer_audios(self, unique_id_list):
        i=0
        old_dataframe = pd.read_sql_table('API_old', columns=['broadcast_id','date'], con=self.engine)
        for values in unique_id_list:
            api_url = self.url_audios + str(values)
            old_dataframe_byID = old_dataframe.loc[old_dataframe['broadcast_id'] == str(values)]
            newest_audio_date = old_dataframe_byID.date.max()
            r = requests.get(api_url)
            json_data = r.json()
            json_data = json_normalize(json_data, ['audio_list'],
                                       ['link_before', 'link_next', 'date_before', 'date_next', 'date'])
            newest_audio = json_data['date'].unique()
            if(newest_audio == newest_audio_date):
                print('Gleiches Datum!')
            elif(len(json_data['date'].unique()) == 0):
                print('Länge der Liste: ', len(json_data['date'].unique()))
            else:
                print('Neuere Audios sind verfügbar!')
                if(i==0):
                    df = pd.DataFrame.from_dict(json_data, orient='columns')
                    link = df['link_next'].tolist()[-1]
                else:
                    df = df.append(pd.DataFrame.from_dict(json_data, orient='columns'), ignore_index=True)
                    link = df['link_next'].tolist()[-1]
                    if (link == ''):
                        continue
                while (link != '' or json_data['date'].unique() != newest_audio_date or i<1):
                    req = requests.get(link)
                    json_data = req.json()
                    json_data = json_normalize(json_data, ['audio_list'],
                                               ['link_before', 'link_next', 'date_before', 'date_next', 'date'])
                    if json_data['date'].unique() == newest_audio_date:
                        print('Same')
                        break
                    new_df = pd.DataFrame.from_dict(json_data, orient='columns')
                    df = df.append(new_df, ignore_index=True)
                    link = df['link_next'].tolist()[-1]
                    if link == '':
                        print('Empty Link!')
                        break
                i = i + 1
        try:
            self.change_dataframe(dataframe=df)
        except:
            print('Dataframe is empty!')
"""


