# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:36:45 2024

@author: Mason
"""
import pandas as pd
import os

df = pd.read_csv('data/train.csv')
parq_files = os.listdir('data/series_train.parquet')

file_ids = []
for idx in range(0, len(parq_files)):
    file_ids.append(parq_files[idx].replace('id=', ''))

series_available = []

for row in df['id']:
    if row in file_ids:
        series_available.append(True)
    else:
        series_available.append(False)
        
df['Series Available'] = series_available

df.drop(df[df['Series Available'] == False].index, inplace=True)

df_nn = df[['id','sii']]
id_list = list(df_nn['id'])
parq_list = []

for file in parq_files:
    path = 'data/series_train.parquet/' + str(file) + '/part-0.parquet'
    parquet = pd.read_parquet(path)
    id_val = file.replace('id=', '')
    id_idx = id_list.index(id_val)
    parq_list.append(parquet)


