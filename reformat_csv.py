"""Developed by: Joseph Pham 2021

This module is solely used to perform simple reformatting operations on CSV files for further use in machine learning
It will continue to be updated as needs arise.

Returns a saved copy of the newly reformatted CSV file, it does not alter the original CSV file.
"""

import csv
import pandas as pd
import os
import sys


def remove_columns(df, cols):
    """Remove a column or list of columns from Dataframe
    Args:
         :param: df (dataframe): pandas dataframe containing a database
         :param: cols: list containing column names to remove
    Returns:
        df (dataframe): removed columns from dataframe
     """
    try:
        df.drop(cols, inplace=True, axis=1)
    except KeyError:
        print("One or more of the columns to remove does not exist in the dataframe.\n")
    return df


def add_columns_from_other_file(other_filename, columns, df):
    """Add a column(s) of data from one csv file into another csv file
    Args:
        :param: other_filename (string): string containing the filepath of the other csv file
        :param: columns (array): list containing 1 or more column names
        :param: df (pandas dataframe): dataframe of the dataset being reformatted
    Returns: dataset with the additional column(s) of data appended from the other csv file
    """
    # read in csv file as a pd df
    other_file = pd.read_csv(other_filename)

    # iterate through columns and append those to dataset
    for col in columns:
        column = other_file[col]
        reformatted_df = df.join(column)

    return reformatted_df


if __name__ == '__main__':
    assert len(sys.argv) == 2, "First command line argument is name of the new CSV database file,"
    output_file = "Reformatted_Files/" + str(sys.argv[1])

    # read the csv file into a Pandas dataframe
    filename = 'Reformatted_Files/reformatted_database.csv'
    dataset = pd.read_csv(filename)

    # remove desired columns, can put any number of columns into this list
    columns_to_delete = ['Bound Fraction', 'Interprot']
    dataset = remove_columns(dataset, columns_to_delete)

    # add desired columns of data from other files into the current dataset being reformatted
    # can input more columns as needed into the following list
    columns_to_add = ['Bound Fraction']
    dataset = add_columns_from_other_file('Reformatted_Files/reformatted_database.csv', columns_to_add, dataset)

    # save CSV file to Reformatted_Files directory with the output file name inputted
    dataset.to_csv(path_or_buf=output_file, header=list(dataset.columns.values), index=False)
    print("Successfully reformatted CSV database file\n")






