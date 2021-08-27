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
         :param: cols (list): list containing column names to remove
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


def replace_columns_from_other_file(other_filename, columns, df):
    """Replaces a column(s) of data from one csv file into another csv file
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
        df.replace(df[col], column, inplace=True)

    return df


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
        print(columns)
    else:
        columns = [input_data.name]
        print("columns: {}".format(columns))

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
    if len(key_errors) > 0:
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
    df.to_csv(pathway, index=False)


def clean_original_data():
    """ Due to the nested structure of the original Excel file, reformat the original Excel file into a more easily
    manipulated dataframe that can be used for other operations like inserting new data by accession number and etc.

    Args: None
    Returns:None
    """
    # create 2 different DataFrames to read and parse different headers
    info1 = pd.read_csv('Input_Files/information.csv', header=[0])
    info2 = pd.read_csv('Input_Files/information.csv', header=[1])

    # drop undesired columns, and preserve condition-independent data into the index
    cols_to_drop = ['identified protein, description', 'gene names', 'Gene Ontology, Biological Process ',
                    'Gene Ontology, Cellular Component', 'Gene Ontology, Molecular Function', 'Unnamed: 40']
    info2.drop(labels=cols_to_drop, axis=1, inplace=True)
    info2.set_index(keys=['accession number', 'pI', 'length', 'molecular weight', 'Enzyme Commission number'], inplace=True)

    # get the various conditions from info1
    conditions = []
    for col in info1.columns:
        if col.find('Unnamed:') == -1:
            conditions.append(col)

    condition_arr = []
    nest = []
    # to create the multi-index, loop through the columns and create the arrays needed to create MultiIndex
    for idx, label in enumerate(info2.columns):
        label += '.' + str(idx)
        split_label = label.split(sep='.')
        del split_label[len(split_label) - 1]

        if len(split_label) > 1:
            char = split_label[len(split_label) - 1]
            condition = conditions[int(char)]
            nest.append(''.join(split_label[0:len(split_label) - 1]))
        else:
            condition = conditions[0]
            nest.append(''.join(split_label))
        condition_arr.append(condition)

    # get rid of any whitespace for nicer formatting
    for i in range(len(nest)):
        nest[i] = nest[i].rstrip()

    # create a new DataFrame with a proper MultiIndex
    arrays = [condition_arr, nest]
    tuples = list(zip(*arrays))
    multi_index = pd.MultiIndex.from_tuples(tuples, names=['conditions', 'values'])
    df = pd.DataFrame(data=info2.values, index=info2.index, columns=multi_index)

    # drop nested non-Bound-Fraction columns
    df.drop(labels=['Bound Fraction, standard deviation in NSAF value', 'Unbound Fraction, average NSAF value'
                    , 'Unbound Fraction, average NSAF value', 'Unbound Fraction, standard deviation in NSAF value'
                    , 'Enrichment Factor'], axis=1, level='values', inplace=True)

    # replace the '---'/0.00E+00 values with 0 in the DataFrame
    df.replace(to_replace=['---', '0.00E+00'], value=0, inplace=True)
    df.reset_index(level=['pI', 'length', 'molecular weight', 'Enzyme Commission number'], inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    # replace 0 values for molecular weight
    df['molecular weight'] = df['molecular weight'].replace(to_replace=0.00, value=np.nan)

    # instead, return the df so it can be used for other operations
    return df


def insert_accession_numbers(inputted_data):
    """ Insert relevant data by accession number into the modeling dataset

    Args:
        pandas DataFrame inputted_data: the reformatted input DataFrame to be used for inserting data into the modeling dataset
    Returns: At the moment, None
    """
    dataset = pd.read_csv('Input_Files/database.csv')
    sliced_input_data = inputted_data.xs('Bound Fraction, average NSAF value', level='values', axis=1)
    conditions_list = list(sliced_input_data.columns)

    # return individual DataFrames for newly inserted data, and then concat that with the original dataframe
    # missing_a_num_df = insert_missing_accession_numbers(list(dataset['Accession Number']), conditions_list, dataset, inputted_data, sliced_input_data)
    incomplete_a_num_df = insert_incomplete_accession_numbers(list(dataset['Accession Number']), conditions_list, dataset, inputted_data, sliced_input_data)

    dataset = pd.concat(objs=[dataset, missing_a_num_df, incomplete_a_num_df], ignore_index=True)
    # dataset.to_csv(path_or_buf='Input_Files/prototype_database.csv')
    print(dataset)


def insert_missing_accession_numbers(accession_numbers, conditions, dataset, data, sliced_data):
    # find accession numbers excluded from modeling dataset
    missing_accession_numbers = set()
    for access_num in list(sliced_data.index):
        if access_num not in accession_numbers:
            missing_accession_numbers.add(access_num)
    print(f'Number of missing accession numbers: {len(missing_accession_numbers)}\n')

    # insert the Bound Fraction values for the missing accession numbers
    missing_a_num_df = pd.DataFrame(columns=dataset.columns)

    # loop through all of the accession numbers in the inputted dataset
    for condition in conditions:
        # filter for particle properties
        split_label = condition.split()
        if split_label[0].find('(+)') != -1:
            particle_charge = 1
        else:
            particle_charge = 0

        if split_label[1] == '10':
            particle_size = 10
        else:
            particle_size = 100

        # filter for solvent conditions
        if len(split_label) <= 9:
            cys_concentration = nacl_concentration = 0
        else:
            # check for cysteine or sodium chloride concentration
            if split_label[12] == 'NaCl':
                nacl_concentration = split_label[10]
                cys_concentration = 0
            else:
                cys_concentration = split_label[10]
                nacl_concentration = 0
        for a_num in missing_accession_numbers:
            missing_a_num_df.at[len(missing_a_num_df.index), 'Particle Charge'] = particle_charge
            missing_a_num_df.at[len(missing_a_num_df.index) - 1, 'Particle Size'] = particle_size

            # inserting solvent + concentration
            missing_a_num_df.at[len(missing_a_num_df.index) - 1, 'Solvent NaCl Concentration'] = nacl_concentration
            missing_a_num_df.at[len(missing_a_num_df.index) - 1, 'Solvent Cysteine Concentration'] = cys_concentration

            # insert other relevant data
            missing_a_num_df.at[len(missing_a_num_df.index) - 1, 'Accession Number'] = a_num
            missing_a_num_df.at[len(missing_a_num_df.index) - 1, 'Bound Fraction'] = sliced_data.at[a_num, condition]
            missing_a_num_df.at[len(missing_a_num_df.index) - 1, 'Enzyme Commission Number'] = data.loc[a_num, 'Enzyme Commission number'].iat[0]
            missing_a_num_df.at[len(missing_a_num_df.index) - 1, 'pI'] = data.loc[a_num, 'pI'].iat[0]
            missing_a_num_df.at[len(missing_a_num_df.index) - 1, 'Protein Length'] = data.loc[a_num, 'length'].iat[0]
            missing_a_num_df.at[len(missing_a_num_df.index) - 1, 'Protein Weight'] = data.loc[a_num, 'molecular weight'].iat[0]

    return missing_a_num_df


def insert_incomplete_accession_numbers(accession_numbers, conditions, dataset, data, sliced_data):
    incomplete_accession_numbers = set()
    occurrences_of_accession_numbers = {}

    # find list of incomplete accession numbers
    for a_num in accession_numbers:
        if a_num not in occurrences_of_accession_numbers:
            occurrences_of_accession_numbers[a_num] = 1
        else:
            occurrences_of_accession_numbers[a_num] += 1

    # filter for incomplete Accession Numbers
    for key in occurrences_of_accession_numbers.keys():
        if occurrences_of_accession_numbers[key] < 6:
            incomplete_accession_numbers.add(key)

    print(f'Number of incomplete accession numbers: {len(incomplete_accession_numbers)}\n')
    incomplete_a_num_df = pd.DataFrame(columns=dataset.columns)

    dataset.set_index(keys='Accession Number', inplace=True)

    base_string = 'AgNP, 10 mM NaPi, pH 7.4'
    condition = ''

    # filter for missing conditions, and insert those if possible (Might be NaN and 0 values?)
    accession_numbers = set()
    for idx in dataset.index:
        # simply pass if the Accession Number has already been accessed
        if idx in accession_numbers:
            pass
        else:
            # track the accession numbers using a set
            accession_numbers.add(idx)

            # slice the dataset by Accession Number
            accession_number = idx
            sliced_df = dataset.loc[idx]

            # if there is only one instance of the Accession Number (returns a pandas Series)
            if len(sliced_df.shape) < 2:
                pass
            else:
                # drop the identical accession numbers in order to use default integer indexing
                sliced_df.reset_index(inplace=True, drop=True)

                # if an Accession Number occurs less than 6 times, check for specific missing conditions
                if sliced_df.shape[0] < 6:
                    # iterate through the rows and construct conditions
                    for row in sliced_df.index:
                        # condition = base_string
                        pass
                    pass
                pass
    # INSERT CODE ABOVE TO FILTER BY INCOMPLETE CONDITIONS #

    sys.exit(0)
    return incomplete_a_num_df


def find_bound_fraction_filler(original):
    # how to go about this:
    sliced_input_data = original.xs('Bound Fraction, average NSAF value', level='values', axis=1)
    print(original)
    print(sliced_input_data)
    dataset = pd.read_csv('Input_Files/database.csv')
    freq = dataset['Bound Fraction'].value_counts()

    # check out the mode and frequency of Bound Fraction values
    # print(freq)
    # print(dataset['Bound Fraction'].mode())

    base_string = 'AgNP, 10 mM NaPi, pH 7.4'
    condition = ''
    possible_filler_bf_values = {}
    for row in dataset.index:
        condition = base_string
        accession_number = dataset.at[row, 'Accession Number']

        # filter out specific solvent conditions in order to construct condition
        if dataset.at[row, 'Solvent NaCl Concentration'] != 0.0:
            condition += ' + ' + str(dataset.at[row, 'Solvent NaCl Concentration']) + ' mM NaCl'
        elif dataset.at[row, 'Solvent Cysteine Concentration'] != 0.0:
            condition += ' + ' + str(dataset.at[row, 'Solvent Cysteine Concentration']) + ' mM cys'

        if dataset.at[row, 'Particle Size'] == 10:
            condition = '10 nm ' + condition
        else:
            condition = '100 nm ' + condition

        if dataset.at[row, 'Particle Charge'] == 1:
            condition = '(+) ' + condition
        else:
            condition = '(-) ' + condition

        # filter for any mismatches and record the Accession Number and associated Bound Fraction value from the
        # modeling dataset
        if dataset.at[row, 'Bound Fraction'] != sliced_input_data.at[accession_number, condition]:
            print(f"\nThe accession number with mismatching values: {accession_number}")
            print(f"database.csv Bound Fraction value: {dataset.at[row, 'Bound Fraction']}")
            print(f"filter.csv Bound Fraction value: {sliced_input_data.at[accession_number, condition]}")
            print(f"The condition at which the value is mismatched: {condition}\n")
            if accession_number not in possible_filler_bf_values.keys():
                possible_filler_bf_values[accession_number] = dataset.at[row, 'Bound Fraction']

    # get the most occurring mismatched value
    print(possible_filler_bf_values)
    print(max(possible_filler_bf_values.values()))


if __name__ == '__main__':
    original_data = clean_original_data()
    # find_bound_fraction_filler(original_data)
    insert_accession_numbers(original_data)


