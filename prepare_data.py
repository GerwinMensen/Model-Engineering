# import python scripts
import data_load as dl
from sklearn.preprocessing import OneHotEncoder

# import packages
import numpy as np
import pandas as pd


def prepare_data(dataframe_transactions):
    dataframe_transactions['month'] = dataframe_transactions["tmsp"].dt.month_name()
    dataframe_transactions['weekday'] = dataframe_transactions["tmsp"].dt.day_name()

    # Aufteilen der Spalten in numerische und kategorische Felder
    X_num = dataframe_transactions.select_dtypes(exclude='object')
    X_cat = dataframe_transactions.select_dtypes(include='object')

    # einen OneHotEncoder erstellen
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_encoded = encoder.fit_transform(X_cat)

    # Spalten benennen
    categorical_columns = [f'{col}_{cat}' for i, col in enumerate(X_cat.columns) for cat in encoder.categories_[i]]

    # numerische Spalten und die gerade erzeugten wieder zusammenführen
    # numerical features
    one_hot_features = pd.DataFrame(X_encoded, columns=categorical_columns)
    dataframe_prepared = X_num.join(one_hot_features)
    dataframe_prepared = dataframe_prepared.join(X_cat)


    # Erste Spalte benennen
    dataframe_prepared.rename(columns={dataframe_transactions.columns[0]: "ID"}, inplace=True)

    # Extra Spalten für die einzelnen Bestandteile des Timestamps erstellen
    dataframe_prepared['date'] = dataframe_prepared["tmsp"].dt.date
    dataframe_prepared['day'] = dataframe_prepared["tmsp"].dt.day
    dataframe_prepared['time'] = dataframe_prepared["tmsp"].dt.time
    dataframe_prepared['hour'] = dataframe_prepared["tmsp"].dt.hour
    dataframe_prepared['minute'] = dataframe_prepared["tmsp"].dt.minute
    dataframe_prepared['second'] = dataframe_prepared["tmsp"].dt.second
    # dataframe_transactions['month_as_int'] = dataframe_transactions["tmsp"].dt.month


    # Gebühr für erfolgreiche Transaktion
    dataframe_prepared['success_fee'] = np.where(dataframe_prepared["PSP_Moneycard"] == 1, 5,
                                                     np.where(dataframe_prepared["PSP_Goldcard"] == 1, 10,
                                                              np.where(dataframe_prepared["PSP_UK_Card"] == 1, 3,1)))
    # Gebühr für fehlgeschlagene Transaktion
    dataframe_prepared['fail_fee'] = np.where(dataframe_prepared["PSP_Moneycard"] == 1, 2,
                                                     np.where(dataframe_prepared["PSP_Goldcard"] == 1, 5,
                                                              np.where(dataframe_prepared["PSP_UK_Card"] == 1, 1,0.5)))


    # temporäre Spalte "tmsp - 1 Minute" erzeugen und anfügen dieser an den Datensatz
    dataframe_prepared['tmsp_minus_one_min'] = dataframe_prepared["tmsp"] - np.timedelta64(1, 'm')


    # Transaktionen nach timestamp sortieren
    dataframe_prepared.sort_values(by=['tmsp'])

    # Anzahl Datensätze bestimmen
    number_of_rows = len(dataframe_prepared)
    # leeres Series-Objekt erstellen, in welchem abgelegt wird, ob zu dem Datensatz ein vorheriger anderer Datensatz
    # gefunden werden konnte, der zum gleichen Einkauf gehört
    has_predecessor = pd.Series([])
    # Spalte für die Anzahl der bisherigen Versuche
    current_number_of_try = pd.Series([])
    # Spalte für die bisher gezahlte Gebühr
    current_paid_fee = pd.Series([])
    # leeres Series-Objekt erstellen, in welchem die Höhe der Gebühr abgelegt wird
    # nach Abschluss der Spalte die binäre Variable, ob ein Einkauf Vorgänger hat, anfügen
    dataframe_prepared.insert(10, 'has_predecessors', has_predecessor)
    dataframe_prepared.insert(10, 'current_number_of_try', current_number_of_try)
    dataframe_prepared.insert(10, 'current_paid_fee', current_paid_fee)
    # jeden Datensatz in einer Schleife durchgehen
    for i in range(0, number_of_rows):
        # Variablen mit den Werten aus dem aktuell betrachteten Datensatz belegen
        actual_country = dataframe_prepared.iloc[i].loc['country']
        actual_amount = dataframe_prepared.iloc[i].loc['amount']
        actual_tmsp = dataframe_prepared.iloc[i].loc['tmsp']
        actual_tmsp_minus_one_min = dataframe_prepared.iloc[i].loc['tmsp_minus_one_min']

        # Den maximalen Vorgänger bestimmen
        predecessor_ID = np.where(
                                        # vorheriger Datensatz muss früher sein, aber nicht mehr als 1 Minute früher
                                        (dataframe_prepared['tmsp'] < actual_tmsp)
                                        & (dataframe_prepared['tmsp'] > actual_tmsp_minus_one_min)
                                        # vorheriger Datensatz muss den gleichen Betrag haben
                                        & (dataframe_prepared['amount'] == actual_amount)
                                        # vorheriger Datensatz muss fehlgeschlagen sein
                                        & (dataframe_prepared['success'] == 0)
                                        # vorheriger Datensatz muss aus dem gleichen Land kommen
                                        & (dataframe_prepared['country'] == actual_country)
                                        , dataframe_prepared['ID'], -1 ).max()
        if predecessor_ID != -1:
            dataframe_prepared.iloc[i, dataframe_prepared.columns.get_loc('has_predecessors')] = 1
            dataframe_prepared.iloc[i, dataframe_prepared.columns.get_loc('current_number_of_try')] \
                = 1 + np.where(dataframe_prepared['ID'] == predecessor_ID,
                               dataframe_prepared['current_number_of_try'], 0 ).max()
            dataframe_prepared.iloc[i, dataframe_prepared.columns.get_loc('current_paid_fee')] \
                = np.where(dataframe_prepared['ID'] == predecessor_ID,
                                             dataframe_prepared['fail_fee'], 0 ).max() \
                  + np.where(dataframe_prepared['ID'] == predecessor_ID,
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











