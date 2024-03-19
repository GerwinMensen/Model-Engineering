# import python scripts
import data_load as dl

# import packages
import numpy as np
import pandas as pd


def prepare_data(dataframe_transactions):
    # Erste Spalte benennen
    dataframe_transactions.rename(columns={dataframe_transactions.columns[0]: "ID"}, inplace=True)

    # Extra Spalten für die einzelnen Bestandteile des Timestamps erstellen
    dataframe_transactions['date'] = dataframe_transactions["tmsp"].dt.date
    dataframe_transactions['day'] = dataframe_transactions["tmsp"].dt.day
    dataframe_transactions['time'] = dataframe_transactions["tmsp"].dt.time
    dataframe_transactions['hour'] = dataframe_transactions["tmsp"].dt.hour
    dataframe_transactions['minute'] = dataframe_transactions["tmsp"].dt.minute
    dataframe_transactions['second'] = dataframe_transactions["tmsp"].dt.second
    dataframe_transactions['month_as_int'] = dataframe_transactions["tmsp"].dt.month
    dataframe_transactions['month_as_str'] = dataframe_transactions["tmsp"].dt.month_name()
    dataframe_transactions['weekday_as_str'] = dataframe_transactions["tmsp"].dt.day_name()
    dataframe_transactions['weekday_as_int'] = dataframe_transactions["tmsp"].dt.weekday
    dataframe_transactions['Is_Monday'] = np.where(dataframe_transactions["tmsp"].dt.weekday == 0, 1, 0)
    dataframe_transactions['Is_Tuesday'] = np.where(dataframe_transactions["tmsp"].dt.weekday == 1, 1, 0)
    dataframe_transactions['Is_Wednesday'] = np.where(dataframe_transactions["tmsp"].dt.weekday == 2, 1, 0)
    dataframe_transactions['Is_Thursday'] = np.where(dataframe_transactions["tmsp"].dt.weekday == 3, 1, 0)
    dataframe_transactions['Is_Friday'] = np.where(dataframe_transactions["tmsp"].dt.weekday == 4, 1, 0)
    dataframe_transactions['Is_Saturday'] = np.where(dataframe_transactions["tmsp"].dt.weekday == 5, 1, 0)
    dataframe_transactions['Is_Sunday'] = np.where(dataframe_transactions["tmsp"].dt.weekday == 6, 1, 0)

    # Extra Spalten für den Zahlungsdienstleister erstellen
    dataframe_transactions['Is_Moneycard'] =  np.where(dataframe_transactions["PSP"] == 'Moneycard', 1, 0)
    dataframe_transactions['Is_Goldcard'] = np.where(dataframe_transactions["PSP"] == 'Goldcard', 1, 0)
    dataframe_transactions['Is_UK_Card'] = np.where(dataframe_transactions["PSP"] == 'UK_Card', 1, 0)
    dataframe_transactions['Is_Simplecard'] = np.where(dataframe_transactions["PSP"] == 'Simplecard', 1, 0)

    # Gebühr für erfolgreiche Transaktion
    dataframe_transactions['success_fee'] = np.where(dataframe_transactions["PSP"] == 'Moneycard', 5,
                                                     np.where(dataframe_transactions["PSP"] == 'Goldcard', 10,
                                                              np.where(dataframe_transactions["PSP"] == 'UK_Card', 3,1)))
    # Gebühr für fehlgeschlagene Transaktion
    dataframe_transactions['fail_fee'] = np.where(dataframe_transactions["PSP"] == 'Moneycard', 2,
                                                     np.where(dataframe_transactions["PSP"] == 'Goldcard', 5,
                                                              np.where(dataframe_transactions["PSP"] == 'UK_Card', 1,0.5)))

    # Extra Spalten für die Kartenart erstellen
    dataframe_transactions['Is_Visa'] = np.where(dataframe_transactions["card"] == 'Visa', 1, 0)
    dataframe_transactions['Is_Diners'] = np.where(dataframe_transactions["card"] == 'Diners', 1, 0)
    dataframe_transactions['Is_Master'] = np.where(dataframe_transactions["card"] == 'Master', 1, 0)

    # Extra Spalten für das Land erstellen
    dataframe_transactions['Is_Germany'] = np.where(dataframe_transactions["country"] == 'Germany', 1, 0)
    dataframe_transactions['Is_Austria'] = np.where(dataframe_transactions["country"] == 'Austria', 1, 0)
    dataframe_transactions['Is_Switzerland'] = np.where(dataframe_transactions["country"] == 'Switzerland', 1, 0)

    # temporäre Spalte "tmsp - 1 Minute" erzeugen und anfügen dieser an den Datensatz
    dataframe_transactions['tmsp_minus_one_min'] = dataframe_transactions["tmsp"] - np.timedelta64(1, 'm')

    # Transaktionen nach timestamp sortieren
    dataframe_transactions.sort_values(by=['tmsp'])

    # Anzahl Datensätze bestimmen
    number_of_rows = len(dataframe_transactions)
    # leeres Series-Objekt erstellen, in welchem abgelegt wird, ob zu dem Datensatz ein vorheriger anderer Datensatz
    # gefunden werden konnte, der zum gleichen Einkauf gehört
    has_predecessor = pd.Series([])
    # Spalte für die Anzahl der bisherigen Versuche
    current_number_of_try = pd.Series([])
    # Spalte für die bisher gezahlte Gebühr
    current_paid_fee = pd.Series([])
    # leeres Series-Objekt erstellen, in welchem die Höhe der Gebühr abgelegt wird
    # nach Abschluss der Spalte die binäre Variable, ob ein Einkauf Vorgänger hat, anfügen
    dataframe_transactions.insert(10, 'has_predecessors', has_predecessor)
    dataframe_transactions.insert(10, 'current_number_of_try', has_predecessor)
    dataframe_transactions.insert(10, 'current_paid_fee', has_predecessor)
    # jeden Datensatz in einer Schleife durchgehen
    for i in range(0, number_of_rows):
        # Variablen mit den Werten aus dem aktuell betrachteten Datensatz belegen
        actual_country = dataframe_transactions.iloc[i].loc['country']
        actual_amount = dataframe_transactions.iloc[i].loc['amount']
        actual_date = dataframe_transactions.iloc[i].loc['date']
        actual_tmsp = dataframe_transactions.iloc[i].loc['tmsp']
        actual_success = dataframe_transactions.iloc[i].loc['success']
        actual_psp = dataframe_transactions.iloc[i].loc['PSP']
        actual_fail_fee = dataframe_transactions.iloc[i].loc['fail_fee']
        actual_tmsp_minus_one_min = dataframe_transactions.iloc[i].loc['tmsp_minus_one_min']

        # Den maximalen Vorgänger bestimmen
        predecessor_ID = np.where(
                                        # vorheriger Datensatz muss früher sein, aber nicht mehr als 1 Minute früher
                                        (dataframe_transactions['tmsp'] < actual_tmsp)
                                        & (dataframe_transactions['tmsp'] > actual_tmsp_minus_one_min)
                                        # vorheriger Datensatz muss den gleichen Betrag haben
                                        & (dataframe_transactions['amount'] == actual_amount)
                                        # vorheriger Datensatz muss fehlgeschlagen sein
                                        & (dataframe_transactions['success'] == 0)
                                        # vorheriger Datensatz muss aus dem gleichen Land kommen
                                        & (dataframe_transactions['country'] == actual_country)
                                        , dataframe_transactions['ID'], -1 ).max()
        if predecessor_ID != -1:
            dataframe_transactions.iloc[i, dataframe_transactions.columns.get_loc('has_predecessors')] = 1
            dataframe_transactions.iloc[i, dataframe_transactions.columns.get_loc('current_number_of_try')] \
                = 1 + np.where(dataframe_transactions['ID'] == predecessor_ID,
                               dataframe_transactions['current_number_of_try'], 0 ).max()
            dataframe_transactions.iloc[i, dataframe_transactions.columns.get_loc('current_paid_fee')] \
                = np.where(dataframe_transactions['ID'] == predecessor_ID,
                                             dataframe_transactions['fail_fee'], 0 ).max() \
                  + np.where(dataframe_transactions['ID'] == predecessor_ID,
                                             dataframe_transactions['current_paid_fee'], 0 ).max()
        else:
            dataframe_transactions.iloc[i, dataframe_transactions.columns.get_loc('has_predecessors')] = 0
            dataframe_transactions.iloc[i, dataframe_transactions.columns.get_loc('current_number_of_try')] = 1
            dataframe_transactions.iloc[i, dataframe_transactions.columns.get_loc('current_paid_fee')] = 0

        """
        # Bestimmen, ob es einen Vorgänger gibt, auf welchen die Kriterien bzgl. desselben Einkaufs zutreffen
        dataframe_transactions.iloc[i,dataframe_transactions.columns.get_loc('has_predecessors')] = np.where(
                                        # vorheriger Datensatz muss früher sein, aber nicht mehr als 1 Minute früher
                                        (dataframe_transactions['tmsp'] < actual_tmsp)
                                        & (dataframe_transactions['tmsp'] > actual_tmsp_minus_one_min)
                                        # vorheriger Datensatz muss den gleichen Betrag haben
                                        & (dataframe_transactions['amount'] == actual_amount)
                                        # vorheriger Datensatz muss aus dem gleichen Land kommen
                                        & (dataframe_transactions['country'] == actual_country)
                                        ,1, 0).max()
        """
        """
        # Höhe der Gebühr nach Tabelle in der Aufgabenstellung festlegen
        if actual_success == 1:
            if actual_psp == 'Moneycard':
                fee[i] = 5
            elif actual_psp == 'Goldcard':
                fee[i] = 10
            elif actual_psp == 'UK_Card':
                fee[i] = 3
            elif actual_psp == 'Simplecard':
                fee[i] = 1
        elif actual_success == 0:
            if actual_psp == 'Moneycard':
                fee[i] = 2
            elif actual_psp == 'Goldcard':
                fee[i] = 5
            elif actual_psp == 'UK_Card':
                fee[i] = 1
            elif actual_psp == 'Simplecard':
                fee[i] = 0.5
        """

    # nach Abschluss der Spalte die Höhe der Gebühr anfügen
    # dataframe_transactions.insert(11, 'fee', fee)
    # Spalte Timestamp Minus eine Minute wird nicht mehr benötigt, daher löschen
    dataframe_transactions = dataframe_transactions.drop(labels='tmsp_minus_one_min', axis=1)

    # Tabelle als csv speichern
    dataframe_transactions.to_csv('Datengrundlage.csv', sep=';')
    return dataframe_transactions











