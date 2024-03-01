# import packages
import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer


import data_load as dl




# Variablen definieren
filename_credit_transactions = r'PSP_Jan_Feb_2019.xlsx'

# Daten einlesen
unprepared_df = dl.data_load(filename=filename_credit_transactions)
unprepared_df.rename(columns={unprepared_df.columns[0]: "ID"}, inplace=True)
# Spalte ID wird nicht mehr, daher löschen
unprepared_df = unprepared_df.drop(labels='ID', axis=1)


unprepared_data_dict = unprepared_df.to_dict(orient='list')
vec = DictVectorizer()
prepared_array = vec.fit_transform(unprepared_data_dict).toarray()