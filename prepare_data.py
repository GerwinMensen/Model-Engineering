# import python scripts
import data_load as dl
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# import packages
import numpy as np
import pandas as pd


def prepare_data(dataframe_transactions):
    # unbedeutende erste Spalte ID löschen
    dataframe_transactions = dataframe_transactions.drop( dataframe_transactions.columns[0], axis=1)
    # doppelte Datensätze löschen
    dataframe_transactions = dataframe_transactions.drop_duplicates()
    # Index neu setzen
    dataframe_transactions.reset_index(inplace=True, drop=True)

    dataframe_prepared = pd.DataFrame([])

    # Monatsspalte erzeugen
    dataframe_transactions['month'] = dataframe_transactions["tmsp"].dt.month_name()
    # Wochentagsspalte erzeugen
    dataframe_transactions['weekday'] = dataframe_transactions["tmsp"].dt.day_name()

    # Extra Spalten für die einzelnen Bestandteile des Timestamps erstellen
    dataframe_prepared['date'] = dataframe_transactions["tmsp"].dt.date
    dataframe_prepared['day'] = dataframe_transactions["tmsp"].dt.day
    dataframe_prepared['time'] = dataframe_transactions["tmsp"].dt.time
    dataframe_prepared['hour'] = dataframe_transactions["tmsp"].dt.hour
    dataframe_prepared['minute'] = dataframe_transactions["tmsp"].dt.minute
    dataframe_prepared['second'] = dataframe_transactions["tmsp"].dt.second

    # Gebühr für erfolgreiche Transaktion
    dataframe_prepared['success_fee'] = np.where(dataframe_transactions["PSP"] == 'Moneycard', 5,
                                                 np.where(dataframe_transactions["PSP"] == 'Goldcard', 10,
                                                          np.where(dataframe_transactions["PSP"] == 'UK_Card', 3, 1)))
    # Gebühr für fehlgeschlagene Transaktion
    dataframe_prepared['fail_fee'] = np.where(dataframe_transactions["PSP"] == 'Moneycard', 2,
                                              np.where(dataframe_transactions["PSP"] == 'Goldcard', 5,
                                                       np.where(dataframe_transactions["PSP"] == 'UK_Card', 1, 0.5)))

    # temporäre Spalte "tmsp - 1 Minute" erzeugen und anfügen dieser an den Datensatz
    dataframe_prepared['tmsp_minus_one_min'] = dataframe_transactions["tmsp"] - np.timedelta64(1, 'm')





    # Aufteilen der Spalten in numerische, zeitliche und kategorische Felder
    X_num = dataframe_transactions.select_dtypes(include='number')
    X_time = dataframe_transactions.select_dtypes(include='datetime')
    X_cat = dataframe_transactions.select_dtypes(include='object')
    # die numerischen und zeitlichen Spalten können so hinzugefügt werden
    dataframe_prepared = dataframe_prepared.join(X_num)
    dataframe_prepared = dataframe_prepared.join(X_time)
    # der Kalendertag, die Stunde, die Minute und die Sekunde müssen auch One-Hot-Encoded werden
    X_cat = X_cat.join(dataframe_prepared[['day', 'hour', 'minute', 'second']])
    # einen OneHotEncoder erstellen und fitten
    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = one_hot_encoder.fit_transform(X_cat)

    # Spalten benennen
    categorical_columns = [f'{col}_{cat}' for i, col in enumerate(X_cat.columns) for cat in one_hot_encoder.categories_[i]]

    # numerische Spalten und die gerade erzeugten One-Hot-Encoding Spalten wieder zusammenführen
    one_hot_features = pd.DataFrame(X_encoded, columns=categorical_columns)
    dataframe_prepared = dataframe_prepared.join(one_hot_features)

    label_encoder = LabelEncoder()
    dataframe_prepared['enc_country'] = label_encoder.fit_transform(X_cat['country'])
    dataframe_prepared['enc_PSP'] = label_encoder.fit_transform(X_cat['PSP'])
    dataframe_prepared['enc_card'] = label_encoder.fit_transform(X_cat['card'])
    dataframe_prepared['enc_weekday'] = label_encoder.fit_transform(X_cat['weekday'])
    dataframe_prepared['enc_month'] = label_encoder.fit_transform(X_cat['month'])

    X_cat = X_cat.drop(labels=['day', 'hour', 'minute', 'second'], axis=1)
    dataframe_prepared = dataframe_prepared.join(X_cat)
    # dataframe_transactions['month_as_int'] = dataframe_transactions["tmsp"].dt.month




    # Transaktionen nach timestamp sortieren
    dataframe_prepared.sort_values(by=['tmsp'])

    # Anzahl Datensätze bestimmen
    number_of_rows = len(dataframe_prepared)
    number_of_cols = len(dataframe_prepared.columns)
    # leeres Series-Objekt erstellen, in welchem abgelegt wird, ob zu dem Datensatz ein vorheriger anderer Datensatz
    # gefunden werden konnte, der zum gleichen Einkauf gehört
    has_predecessor = pd.Series([])
    # Spalte für die Anzahl der bisherigen Versuche
    current_number_of_try = pd.Series([])
    # Spalte für die bisher gezahlte Gebühr
    current_paid_fee = pd.Series([])
    # leeres Series-Objekt erstellen, in welchem die Höhe der Gebühr abgelegt wird
    # nach Abschluss der Spalte die binäre Variable, ob ein Einkauf Vorgänger hat, anfügen
    dataframe_prepared.insert(number_of_cols, 'has_predecessors', has_predecessor)
    number_of_cols = len(dataframe_prepared.columns)
    dataframe_prepared.insert(number_of_cols , 'current_number_of_try', current_number_of_try)
    number_of_cols = len(dataframe_prepared.columns)
    dataframe_prepared.insert(number_of_cols, 'current_paid_fee', current_paid_fee)
    # jeden Datensatz in einer Schleife durchgehen
    for i in range(0, number_of_rows):
        # Variablen mit den Werten aus dem aktuell betrachteten Datensatz belegen
        actual_country = dataframe_prepared.iloc[i].loc['country']
        actual_amount = dataframe_prepared.iloc[i].loc['amount']
        actual_tmsp = dataframe_prepared.iloc[i].loc['tmsp']
        actual_tmsp_minus_one_min = dataframe_prepared.iloc[i].loc['tmsp_minus_one_min']

        # Den maximalen Vorgänger bestimmen
        predecessor_tmsp = np.where(
                                        # vorheriger Datensatz muss früher sein, aber nicht mehr als 1 Minute früher
                                        (dataframe_prepared['tmsp'] < actual_tmsp)
                                        & (dataframe_prepared['tmsp'] > actual_tmsp_minus_one_min)
                                        # vorheriger Datensatz muss den gleichen Betrag haben
                                        & (dataframe_prepared['amount'] == actual_amount)
                                        # vorheriger Datensatz muss fehlgeschlagen sein
                                        & (dataframe_prepared['success'] == 0)
                                        # vorheriger Datensatz muss aus dem gleichen Land kommen
                                        & (dataframe_prepared['country'] == actual_country)
                                        , dataframe_prepared['tmsp'].values.astype("float64"), -1 ).max()
        if predecessor_tmsp != -1:
            dataframe_prepared.iloc[i, dataframe_prepared.columns.get_loc('has_predecessors')] = 1
            dataframe_prepared.iloc[i, dataframe_prepared.columns.get_loc('current_number_of_try')] \
                = 1 + np.where(dataframe_prepared['tmsp'].values.astype("float64") == predecessor_tmsp,
                               dataframe_prepared['current_number_of_try'], 0 ).max()
            dataframe_prepared.iloc[i, dataframe_prepared.columns.get_loc('current_paid_fee')] \
                = np.where(dataframe_prepared['tmsp'].values.astype("float64") == predecessor_tmsp,
                                             dataframe_prepared['fail_fee'], 0 ).max() \
                  + np.where(dataframe_prepared['tmsp'].values.astype("float64") == predecessor_tmsp,
                                             dataframe_prepared['current_paid_fee'], 0 ).max()
        else:
            dataframe_prepared.iloc[i, dataframe_prepared.columns.get_loc('has_predecessors')] = 0
            dataframe_prepared.iloc[i, dataframe_prepared.columns.get_loc('current_number_of_try')] = 1
            dataframe_prepared.iloc[i, dataframe_prepared.columns.get_loc('current_paid_fee')] = 0

    # Spalte Timestamp Minus eine Minute und country wird nicht mehr benötigt, daher löschen
    dataframe_prepared = dataframe_prepared.drop(labels=['tmsp_minus_one_min'], axis=1)

    # Tabelle als csv speichern
    dataframe_prepared.to_csv('Datengrundlage.csv', sep=';')
    return dataframe_prepared



class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X), self







