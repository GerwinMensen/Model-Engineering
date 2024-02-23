# import python scripts
import data_load as dl

# import packages
import math
import numpy as np
import pandas as pd
import featuretools as ft
import datetime
"""
def max_possible(group):
...     if group['possible'].any():
...        total = group[ group['possible'] ]['total'].idxmax()
...     else:
...        total = group['total'].idxmax()
...     return group.loc[total]

df.groupby(['col', 'row', 'year'], sort=False).apply(max_possible)
"""



# declare variables
filename_credit_transactions = r'PSP_Jan_Feb_2019.xlsx'


# read data
credit_transactions_df = dl.data_load(filename=filename_credit_transactions)



# print(credit_transactions_df.head(20))
print(credit_transactions_df.dtypes)

# print(credit_transactions_df.country.value_counts())
# print(credit_transactions_df.PSP.value_counts())
# print(credit_transactions_df.success.value_counts())
print(credit_transactions_df.columns)
# print(credit_transactions_df['3D_secured'].value_counts())

day_series = credit_transactions_df["tmsp"].dt.date
time_series = credit_transactions_df["tmsp"].dt.time

# tmsp - 1 Minute
tmsp_minus_one_min = credit_transactions_df["tmsp"] - np.timedelta64(1,'m')
time_series = credit_transactions_df["tmsp"].dt.time
credit_transactions_df.insert(8, 'day', day_series)
credit_transactions_df.insert(9, 'time', time_series)
credit_transactions_df.insert(10, 'tmsp_minus_one_min', tmsp_minus_one_min)



number_of_rows = len(credit_transactions_df)
has_predecessor = pd.Series([])
fee = pd.Series([])

for i in range(0, number_of_rows):
    actual_country = credit_transactions_df.iloc[i].loc['country']
    actual_amount = credit_transactions_df.iloc[i].loc['amount']
    actual_day = credit_transactions_df.iloc[i].loc['day']
    actual_tmsp = credit_transactions_df.iloc[i].loc['tmsp']
    actual_success = credit_transactions_df.iloc[i].loc['success']
    actual_psp = credit_transactions_df.iloc[i].loc['PSP']
    actual_tmsp_minus_one_min = credit_transactions_df.iloc[i].loc['tmsp_minus_one_min']
    number_of_predecessors = len(credit_transactions_df.loc[ (credit_transactions_df['country'] == actual_country)
                                        & (credit_transactions_df['amount'] == actual_amount)
                                        & (credit_transactions_df['day'] == actual_day)
                                        & (credit_transactions_df['tmsp'] < actual_tmsp)
                                        & (credit_transactions_df['tmsp'] > actual_tmsp_minus_one_min)])
    if number_of_predecessors > 0:
        has_predecessor[i] = 1
    else:
        has_predecessor[i] = 0

    if actual_success == 1:
        if actual_psp == 'Moneycard':
            fee[i] = 5
        elif actual_psp == 'Goldcard':
            fee[i] = 10
        elif  actual_psp == 'UK_Card':
            fee[i] = 3
        elif actual_psp == 'Simplecard':
            fee[i] = 1
    elif actual_success == 0:
        if actual_psp == 'Moneycard':
            fee[i] = 2
        elif actual_psp == 'Goldcard':
            fee[i] = 5
        elif  actual_psp == 'UK_Card':
            fee[i] = 1
        elif actual_psp == 'Simplecard':
            fee[i] = 0.5

credit_transactions_df.insert(10, 'has_predecessors', has_predecessor)
credit_transactions_df.insert(11, 'fee', fee)
credit_transactions_df.drop(labels='tmsp_minus_one_min', axis=1)

print(credit_transactions_df)
    # max_value = credit_transactions_df.tmsp.max()
    # print(actual_country)
    # np.where(credit_transactions_df.duplicated() == True, max(df['percentage']), 0)


# print(result)
# print(credit_transactions_df.info)


# feature_matrix_customers, feature_defs = ft.dfs(dataframes=credit_transactions_df)
# print(feature_matrix_customers)
# data = ft.demo.load_mock_customer()
# customers_df = data["customers"]
# print(customers_df)
