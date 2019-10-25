import requests
import pandas as pd
from pandas.io.json import json_normalize
import json
import numpy as np
from ast import literal_eval
import os
from sqlalchemy import create_engine

url='https://srv.deutschlandradio.de/i.2745.de.rpc'

r = requests.get(url)
json_data = r.json()
all_broadcasts_df = pd.DataFrame.from_dict(json_data, orient='columns')
all_broadcasts_df['broadcast_id'] = pd.to_numeric(all_broadcasts_df['broadcast_id'])
all_broadcast_ids = all_broadcasts_df['broadcast_id'].unique().tolist()

print(all_broadcasts_df.head())
print(all_broadcast_ids)

def change_dataframe(dataframe):
    dataframe.dropna(subset=['article'], inplace=True)
    dataframe = dataframe.join(dataframe.article.apply(pd.Series))
    dataframe.to_csv('titatest.csv', index=False)

dataframe = pd.DataFrame()

for values in all_broadcast_ids:
    api_url = 'https://srv.deutschlandradio.de/i.2744.de.rpc?drau:broadcast_id=' + str(values)
    print('ID:', values)
    r = requests.get(api_url)
    json_data = r.json()
    json_data = json_normalize(json_data, ['audio_list'],
                               ['link_before', 'link_next', 'date_before', 'date_next', 'date'])
    df = pd.DataFrame.from_dict(json_data, orient='columns')
    dataframe = dataframe.append(df, ignore_index=True)
    if(len(df != 0 )):
        link = dataframe['link_next'].tolist()[-1]
        if (link != ''):
            while(link != ''):
                req = requests.get(link)
                json_data = req.json()
                json_data = json_normalize(json_data, ['audio_list'],
                                   ['link_before', 'link_next', 'date_before', 'date_next', 'date'])
                new_df = pd.DataFrame.from_dict(json_data, orient='columns')
                dataframe = dataframe.append(new_df, ignore_index=True)
                link = dataframe['link_next'].tolist()[-1]
                #print(link)
                print(len(dataframe))
        else:
            print('Kein Datum vorhanden!')
    else:
        print('Leerer Dataframe!')

"""
print(dataframe.columns)
print(dataframe.article)
"""

change_dataframe(dataframe=dataframe)
