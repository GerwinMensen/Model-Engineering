"""
By means of this function the csv or xls files are loaded and checked whether the files meet the expected format
"""
# import packages
import pandas as pd
import codecs

# create function to load the data
def data_load(filename: str, delimiter=';'):
    if filename[-3:] != 'csv' and filename[-4:] != 'xlsx' and filename[-4:] != 'xlsm' and filename[-3:] != 'xls':
        raise ValueError('No valid filetype')
    # open the file in read mode
    with codecs.open(filename, 'r', encoding='utf-8') as my_file:
        if filename[-3:] == 'csv':
            # read the content of the file and save it in a dataframe
            my_file_content = pd.read_csv(filename, sep=delimiter)
            # close the file
        else:
            my_file_content = pd.read_excel(filename)
    my_file.close()
    return my_file_content