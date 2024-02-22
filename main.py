# import python scripts
import data_load as dl

# import packages
import math
import numpy as np
import pandas as pd
import featuretools as ft

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



print(credit_transactions_df.head(20))
print(credit_transactions_df.dtypes)

print(credit_transactions_df.country.value_counts())
print(credit_transactions_df.PSP.value_counts())
print(credit_transactions_df.success.value_counts())
print(credit_transactions_df.columns)
print(credit_transactions_df['3D_secured'].value_counts())

number_of_rows = len(credit_transactions_df)
# for i in range(0, number_of_rows-1)
#     actual_country = credit_transactions_df[i].country
#    np.where(credit_transactions_df.duplicated() == True, max(df['percentage']), 0)

result = credit_transactions_df.groupby(level="country").max()

print(credit_transactions_df.info)


# feature_matrix_customers, feature_defs = ft.dfs(dataframes=credit_transactions_df)
# print(feature_matrix_customers)
# data = ft.demo.load_mock_customer()
# customers_df = data["customers"]
# print(customers_df)
