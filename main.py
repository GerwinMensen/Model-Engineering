# import python scripts
import data_load as dl

# import packages
import math
import numpy as np
import pandas as pd


# declare variables
filename_credit_transactions = r'PSP_Jan_Feb_2019.xlsx'


# read data
credit_transactions = dl.data_load(filename=filename_credit_transactions)

print(credit_transactions)