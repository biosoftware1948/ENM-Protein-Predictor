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
        print("One of the columns to remove does not exist in the dataframe.\n")
    return df


if __name__ == '__main__':
    assert len(sys.argv) == 2, "First command line argument is name of the new CSV database file, " \

    output_file = "Reformatted_Files/" + str(sys.argv[1])
    print("Output_File: " + output_file + "\n")

    # read the csv file into a Pandas dataframe
    filename = 'Reformatted_Files/reformatted_database.csv'
    # dataset = pd.read_csv(filename)
    dataset2 = pd.read_csv(filename)

    # columns_to_delete = ['Bound Fraction']
    # dataset = remove_columns(dataset, columns_to_delete)

    columns_to_delete2 = ['Bound Fraction', 'Interprot']
    dataset2 = remove_columns(dataset2, columns_to_delete2)

    # dataset.to_csv(path_or_buf=output_file, header=list(dataset.columns.values), index=False)
    dataset2.to_csv(path_or_buf=output_file, header=list(dataset2.columns.values), index=False)
    print("Successfully reformatted CSV database file\n")






