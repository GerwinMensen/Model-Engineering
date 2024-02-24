# import python scripts
import data_load as dl

# import packages
import numpy as np
import pandas as pd

# Variablen definieren
filename_credit_transactions = r'PSP_Jan_Feb_2019.xlsx'

# Daten einlesen
credit_transactions_df = dl.data_load(filename=filename_credit_transactions)
# Erste Spalte benennen
credit_transactions_df.rename(columns={credit_transactions_df.columns[0]: "ID"}, inplace=True)

# Extra Spalten für Tag, Uhrzeit, Monat und Wochentag schaffen und diese am Datensatz anfügen
credit_transactions_df['day'] = credit_transactions_df["tmsp"].dt.date
credit_transactions_df['time'] = credit_transactions_df["tmsp"].dt.time
credit_transactions_df['month'] = credit_transactions_df["tmsp"].dt.month
credit_transactions_df['weekday'] = credit_transactions_df["tmsp"].dt.day_name()
# temporäre Spalte "tmsp - 1 Minute" erzeugen und anfügen dieser an den Datensatz
credit_transactions_df['tmsp_minus_one_min'] = credit_transactions_df["tmsp"] - np.timedelta64(1, 'm')

# Anzahl Datensätze bestimmen
number_of_rows = len(credit_transactions_df)
# leeres Series-Objekt erstellen, in welchem abgelegt wird, ob zu dem Datensatz ein vorheriger anderer Datensatz
# gefunden werden konnte, der zum gleichen Einkauf gehört
has_predecessor = pd.Series([])
# leeres Series-Objekt erstellen, in welchem die Höhe der Gebühr abgelegt wird
fee = pd.Series([])

# jeden Datensatz in einer Schleife durchgehen
for i in range(0, number_of_rows):
    # Variablen mit den Werten aus dem aktuell betrachteten Datensatz belegen
    actual_country = credit_transactions_df.iloc[i].loc['country']
    actual_amount = credit_transactions_df.iloc[i].loc['amount']
    actual_day = credit_transactions_df.iloc[i].loc['day']
    actual_tmsp = credit_transactions_df.iloc[i].loc['tmsp']
    actual_success = credit_transactions_df.iloc[i].loc['success']
    actual_psp = credit_transactions_df.iloc[i].loc['PSP']
    actual_tmsp_minus_one_min = credit_transactions_df.iloc[i].loc['tmsp_minus_one_min']
    # Anzahl der Vorgänger bestimmen, auf welche das Kriterium bzgl. desselben Einkaufs zutrifft
    number_of_predecessors = len(credit_transactions_df.loc[
                                        # vorheriger Datensatz muss früher sein, aber nicht mehr als 1 Minute früher
                                        (credit_transactions_df['tmsp'] < actual_tmsp)
                                        & (credit_transactions_df['tmsp'] > actual_tmsp_minus_one_min)
                                        # vorheriger Datensatz muss am selben Tag sein
                                        & (credit_transactions_df['day'] == actual_day)
                                        # vorheriger Datensatz muss den gleichen Betrag haben
                                        & (credit_transactions_df['amount'] == actual_amount)
                                        # vorheriger Datensatz muss aus dem gleichen Land kommen
                                        & (credit_transactions_df['country'] == actual_country)
                                        ])
    # binäre Variable belegen
    if number_of_predecessors > 0:
        has_predecessor[i] = 1
    else:
        has_predecessor[i] = 0

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

# nach Abschluss der Spalte die binäre Variable, ob ein Einkauf Vorgänger hat, anfügen
credit_transactions_df.insert(10, 'has_predecessors', has_predecessor)
# nach Abschluss der Spalte die Höhe der Gebühr anfügen
credit_transactions_df.insert(11, 'fee', fee)
# Spalte Timestamp Minus eine Minute wird nicht mehr benötigt, daher löschen
credit_transactions_df = credit_transactions_df.drop(labels='tmsp_minus_one_min', axis=1)

# Tabelle als csv speichern
credit_transactions_df.to_csv('Datengrundlage.csv', sep=';')
print(credit_transactions_df)
