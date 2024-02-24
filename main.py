# import python scripts
import data_load as dl
import prepare_data as prepdata


# import packages
import numpy as np
import pandas as pd
from sklearn import svm

# Variablen definieren
filename_credit_transactions = r'PSP_Jan_Feb_2019.xlsx'

# Daten einlesen
unprepared_df = dl.data_load(filename=filename_credit_transactions)
# Daten aufbereiten
prepared_df = prepdata.prepare_data(unprepared_df)

# prepared_df = dl.data_load(filename='Datengrundlage.csv')
# prepared_df = prepared_df.drop(columns='Unnamed: 0', axis=1)

print(prepared_df)
# Unterteilen der vorbereiten Daten nach abh. und unabhängiger Variablen
# Zunächst soll die Erfolgswahrscheinlichkeit prognostiziert werden --> Spalte success
x = prepared_df.drop(columns='success', axis=1)
# nur die numerischen Spalten behalten
x = x.select_dtypes(include=np.number)
y = prepared_df.success

print(x)
print(y)
clf = svm.SVC()

clf.fit(x, y)

