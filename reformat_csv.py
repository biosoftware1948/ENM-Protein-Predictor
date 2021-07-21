"""Developed by: Joseph Pham 2021

This module is solely used to perform simple reformatting operations on CSV files for further use in machine learning
It will continue to be updated as needs arise.

Returns a saved copy of the newly reformatted CSV file, it does not alter the original CSV file.
"""

import csv
import pandas as pd
import numpy as np
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


def add_data_by_accession_number(df, input_data):
    """Insert data from one dataframe input to another based on the associated label with that data
    It inserts data by Accession Number, and both the df and the input should have their respective associated
    'Accession Number' data, so that this code can filter and insert data correctly
    Args:
        :param: df (pandas DataFrame): the dataset which we want to insert new data into
        :param: input (pandas Series or Dataframe): the data we want to input into the dataset
    Returns: the new 'df' with the appended data from 'input'
    """
    # set the accession numbers to indexes to use for filtering
    df.set_index('Accession Number', inplace=True)
    input_data.set_index('Accession Number', inplace=True)

    # obtain the feature(s) whose data is/are being inputted
    if isinstance(input_data, pd.DataFrame):
        columns = list(input_data.columns)
    else:
        columns = [input_data.name]

    for col in columns:
        df[col] = np.nan

    # loop through the Accession Numbers of the dataset receiving the data, and insert data after filtering
    # by Accession Number
    key_errors = []
    for accession_number in df.index:
        for feature in columns:
            try:
                df.at[accession_number, feature] = input_data.loc[accession_number, feature]
            except KeyError:
                if accession_number not in key_errors:
                    key_errors.append(accession_number)
                pass

    # print information about any possible missing Accession Numbers from input_data
    print("Number of missing Accession Numbers: \n{}\n".format(len(key_errors)))
    print("List of missing Accession Numbers themselves: \n{}\n".format(key_errors))

    # reset the indexing of the dataset and return final DataFrame
    df.reset_index(inplace=True)
    return df


def remove_strings(df, column, str_to_delete):
    """Strips any unnecessary characters/strings from a column(s)
    of data, such as '%', or 'ph', and etc., and returns just the data itself
    Args:
        :param: df (Pandas DataFrame): this is the df containing the column to edit
        :param: column (string): the column to edit
        :param: str_to_delete: string or character to delete from the column
    Returns:
        :return: df (Pandas DataFrame): the dataframe containing the column with the str/char removed
    """
    df[column] = df[column].map(lambda x: x.lstrip(str_to_delete).rstrip(str_to_delete))
    return df


def save_to_csv(df, pathway):
    """For convenience, save the DataFrame as a CSV file into whatever pathway is designated
    Args:
        :param: df (Pandas DataFrame): the dataframe that we want to save as a CSV file
        :param: pathway (string): a string defining the file pathway where the CSV file is saved
    Returns: None
    """
    df.to_csv(pathway)


if __name__ == '__main__':
    # note for future self:
    # might want to type in code to receive input on which commands and what you want to edit
    # this will probably take some time to code up

    # This bit of code reformats the dataset to delete and add columns, but this section of code is
    # inflexible since every single command must be hardcoded (will require future revision for ease of use)
    # assert len(sys.argv) == 2, "First command line argument is name of the new CSV database file,"
    # output_file = "Reformatted_Files/" + str(sys.argv[1])
    #
    # # read the csv file into a Pandas dataframe
    # filename = 'Reformatted_Files/reformatted_database.csv'
    # dataset = pd.read_csv(filename)
    #
    # # remove desired columns, can put any number of columns into this list
    # columns_to_delete = ['Bound Fraction', 'Interprot']
    # dataset = remove_columns(dataset, columns_to_delete)
    #
    # # add desired columns of data from other files into the current dataset being reformatted
    # # can input more columns as needed into the following list
    # columns_to_add = ['Bound Fraction']
    # dataset = add_columns_from_other_file('Reformatted_Files/reformatted_database.csv', columns_to_add, dataset)
    #
    # # save CSV file to Reformatted_Files directory with the output file name inputted
    # dataset.to_csv(path_or_buf=output_file, header=list(dataset.columns.values), index=False)
    # print("Successfully reformatted CSV database file\n")

    pd.set_option('display.max_rows', None, 'display.max_columns', None)

    # read in the CSV file with the new features and remove the '%' from h-bonding column
    new_features = pd.read_csv('Reformatted_Files/New_Features.csv')
    new_features = remove_strings(new_features, '% H-bond/polar AA', '%')

    # adds data according to accession numbers (GRAVY scores + % h-bonding)
    original_dataset = pd.read_csv('Reformatted_Files/_new_database.csv')
    new_df = add_data_by_accession_number(original_dataset, new_features)
    new_df.to_csv('Reformatted_Files/_updated_database.csv')











