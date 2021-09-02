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


def clean_original_data():
    """ Due to the nested structure of the original Excel file, reformat the original Excel file into a more easily
    manipulated dataframe that can be used for other operations like inserting new data by accession number and etc.

    Args: None
    Returns:None
    """
    # create 2 different DataFrames to read and parse different headers
    conditions_df = pd.read_csv('Input_Files/InputData.csv', header=[0])
    non_conditions_df = pd.read_csv('Input_Files/InputData.csv', header=[1])

    # drop any unnecessary columns from the dataset
    non_conditions_df.drop(labels=['molecular weight', 'Gene Ontology, Biological Process',
                                   'Gene Ontology, Cellular Component',
                                   'Gene Ontology, Molecular Function'], axis=1, inplace=True)

    # print(non_conditions_df.columns)

    # rename the columns for ease of readability
    non_conditions_df.rename(columns={
        'accession number': 'Accession Number', 'calc. pI': 'pI', '# metal bonds': '# of Metal Bonds',
        '% aromatic': '% Aromatic', 'h-bonding %': '% H-bond/polar AA', '(-) Charge %': '% Negative',
        '(+) charge %': '% Positive', 'length': 'Protein Length', 'Database MW': 'Protein Weight',
        'Enzyme Commission number': 'Enzyme Commission Number'
    }, inplace=True)

    # isolate the solvent conditions
    conditions = []
    for col in conditions_df.columns:
        if col.find('Unnamed:') == -1:
            conditions.append(col)

    # isolate the condition_independent data
    condition_independent_labels = ['Accession Number', 'pI', '# of Metal Bonds', '% Aromatic', '% H-bond/polar AA',
                                    '% Negative', '% Positive', 'Protein Length', 'Protein Weight', 'Sequence',
                                    'GRAVY', 'Enzyme Commission Number', '% Cysteine', '% Hydrophilic']
    condition_independent_df = non_conditions_df[condition_independent_labels]

    non_conditions_df.drop(labels=condition_independent_labels, axis=1, inplace=True)

    condition_arr = []
    nest = []
    # to create the multi-index, loop through the columns and create the arrays needed to create MultiIndex
    for idx, label in enumerate(non_conditions_df.columns):
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
    df = pd.DataFrame(data=non_conditions_df.values, index=non_conditions_df.index, columns=multi_index)

    # drop nested non-Bound-Fraction columns
    df.drop(labels=['Bound Fraction, standard deviation in NSAF value', 'Unbound Fraction, average NSAF value',
                    'Unbound Fraction, average NSAF value', 'Unbound Fraction, standard deviation in NSAF value',
                    'Enrichment Factor'], axis=1, level='values', inplace=True)

    # currently turns the MultiIndex into tuples, but that will suffice for inserting new information and Bound Fraction
    df = pd.concat(objs=[condition_independent_df, df], axis=1)

    # clean out % symbols from % data features (i.e. % Cysteine)
    cols_to_delete_chars = ['% Aromatic', '% H-bond/polar AA', '% Negative', '% Positive', '% Cysteine', '% Hydrophilic']
    for col in cols_to_delete_chars:
        remove_strings(df, col, '%')

    # pd.set_option('display.max_columns', None, 'display.max_rows', None)
    # replace the '---'/0.00E+00 values with 0 in the DataFrame for Bound Fraction values
    df.replace(to_replace=['---', '0.00E+00'], value=0, inplace=True)

    # convert datatypes to either float or int, and preserve non-numerical data
    df['Protein Weight'] = df['Protein Weight'].apply(lambda x: int(x.replace(',', '')))
    df.set_index(keys=['Sequence', 'Accession Number'], inplace=True)
    df = df.apply(pd.to_numeric, downcast='integer', errors='coerce')
    df.reset_index(level=['Sequence'], inplace=True)
    return df


# rename Bound Fraction columns
def rename(col):
    if isinstance(col, tuple):
        col = '_'.join(str(c) for c in col)
        col = col.rstrip('Bound Fraction, average NSAF value').rstrip('_')
    return col


def insert_accession_numbers(inputted_data):
    """ Insert relevant data by accession number into the modeling dataset

    Args:
        pandas DataFrame inputted_data: the reformatted input DataFrame to be used for inserting data into the modeling dataset
    Returns: At the moment, None
    """
    dataset = pd.read_csv('Input_Files/database.csv')

    # rename columns
    inputted_data.columns = map(rename, inputted_data.columns)

    # get all solvent conditions
    conditions_list = ['(-) 10 nm AgNP, 10 mM NaPi, pH 7.4', '(-) 100 nm AgNP, 10 mM NaPi, pH 7.4',
                       '(-) 10 nm AgNP, 10 mM NaPi, pH 7.4 + 0.1 mM cys',
                       '(-) 10 nm AgNP, 10 mM NaPi, pH 7.4 + 0.8 mM NaCl',
                       '(-) 10 nm AgNP, 10 mM NaPi, pH 7.4 + 3.0 mM NaCl']

    # return individual DataFrames for newly inserted data, and then concat that with the original dataframe
    # missing_a_num_df = insert_missing_accession_numbers(list(dataset['Accession Number']), conditions_list, dataset, inputted_data)
    incomplete_a_num_df = insert_incomplete_accession_numbers(list(dataset['Accession Number']), conditions_list, dataset, inputted_data)
    sys.exit(0)
    dataset.reset_index(inplace=True)

    # join the DataFrames together to create a comprehensive dataset and save it as a CSV file
    dataset = pd.concat(objs=[dataset, missing_a_num_df, incomplete_a_num_df], ignore_index=True)
    # dataset.to_csv(path_or_buf='Input_Files/prototype_database.csv', index=False)


def insert_missing_accession_numbers(accession_numbers, conditions, dataset, data):
    """
    Args:
        list accession_numbers:
        list conditions:
        pandas DataFrame dataset:
        pandas DataFrame data:
        pandas DataFrame sliced_data:
    Returns:

    """
    # find accession numbers excluded from modeling dataset
    missing_accession_numbers = set()
    for access_num in data.index:
        if access_num not in accession_numbers:
            missing_accession_numbers.add(access_num)
    print(f'Number of missing accession numbers: {len(missing_accession_numbers)}\n')

    # insert the Bound Fraction values for the missing accession numbers
    missing_a_num_df = pd.DataFrame(columns=dataset.columns)

    # loop through all of the accession numbers in the inputted dataset
    for condition in conditions:
        # grab positive or negative charge
        split_label = str(condition).lstrip('(').rstrip(')').replace("'", '').split()
        if split_label[0].find('(+)') != -1:
            particle_charge = 1
        else:
            particle_charge = 0

        # grab particle size
        if split_label[1] == '10':
            particle_size = 10
        else:
            particle_size = 100

        # filter for solvent conditions
        if len(split_label) <= 14:
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
            last_row_index = len(missing_a_num_df.index)
            missing_a_num_df.at[last_row_index, 'Particle Charge'] = particle_charge
            missing_a_num_df.at[last_row_index, 'Particle Size'] = particle_size

            # inserting solvent + concentration
            missing_a_num_df.at[last_row_index, 'Solvent NaCl Concentration'] = nacl_concentration
            missing_a_num_df.at[last_row_index, 'Solvent Cysteine Concentration'] = cys_concentration
            missing_a_num_df.at[last_row_index, 'Accession Number'] = a_num

            for feature in dataset.columns:
                # ignore certain features if they've been inserted or are simply unneeded
                if feature == 'Particle Charge' or feature == 'Particle Size' or \
                   feature == 'Solvent NaCl Concentration' or feature == 'Solvent Cysteine Concentration' or \
                   feature == 'Accession Number' or feature == 'Enrichment':
                    pass
                # requires passing in a special condition string to access Bound Fraction in the InputData sheet
                elif feature == 'Bound Fraction':
                    missing_a_num_df.at[last_row_index, feature] = data.at[a_num, condition]
                else:
                    missing_a_num_df.at[last_row_index, feature] = data.at[a_num, feature]
    return missing_a_num_df


def insert_incomplete_accession_numbers(accession_numbers, conditions, dataset, data):
    """ Insert relevant data by missing protein particle conditions by Accession Number

    Args:
        list accession_numbers: list of accession numbers in the dataset
        list conditions: list of protein particle conditions
        pandas DataFrame dataset: dataframe containing the modeling dataset to insert information into
        pandas DataFrame data: dataframe containing the original Excel file to insert information from
        pandas DataFrame sliced_data: dataframe from slicing the original Excel file, which isolates the Bound Fraction information
    Returns: At the moment, None
    """
    incomplete_accession_numbers = set()
    occurrences_of_accession_numbers = {}

    # find list of incomplete accession numbers
    for accession_number in accession_numbers:
        if accession_number not in occurrences_of_accession_numbers:
            occurrences_of_accession_numbers[accession_number] = 1
        else:
            occurrences_of_accession_numbers[accession_number] += 1

    # filter for incomplete Accession Numbers
    for key in occurrences_of_accession_numbers.keys():
        if occurrences_of_accession_numbers[key] < 6:
            incomplete_accession_numbers.add(key)

    print(f'Number of incomplete accession numbers: {len(incomplete_accession_numbers)}\n')
    incomplete_a_num_df = pd.DataFrame(columns=dataset.columns)
    dataset.set_index(keys='Accession Number', inplace=True)
    base_string = "AgNP, 10 mM NaPi, pH 7.4"
    condition = ''

    # filter for missing conditions, and insert those if possible (Might be NaN and 0 values?)
    accession_numbers = set()
    accessed_conditions = []
    for idx in dataset.index:
        # simply pass if the Accession Number has already been accessed
        if idx in accession_numbers:
            pass
        else:
            # track the accession numbers using a set
            accession_numbers.add(idx)
            accessed_conditions = []

            # slice the dataset by Accession Number
            accession_number = idx
            sliced_df = dataset.loc[idx]

            # if there is only one instance of the Accession Number (returns a pandas Series)
            if len(sliced_df.shape) < 2:
                # INSERT CODE TO DEAL WITH SIMPLE SERIES #
                # construct the solvent conditions based on the accessible values within the Series
                # print(sliced_df['Solvent Cysteine Concentration'])
                pass
            else:
                # drop the identical accession numbers in order to use default integer indexing
                sliced_df.reset_index(inplace=True, drop=True)

                # if an Accession Number occurs less than 6 times, check for specific missing conditions
                if sliced_df.shape[0] < 6:
                    # accessed_conditions = []
                    # iterate through the rows and construct conditions
                    for row in sliced_df.index:
                        condition = base_string

                        # add in particle size
                        if sliced_df.at[row, 'Particle Size'] == 10:
                            condition = '10 nm ' + condition
                        else:
                            condition = '100 nm ' + condition

                        # add in particle charge
                        if sliced_df.at[row, 'Particle Charge'] == 1:
                            condition = '(+) ' + condition
                        else:
                            condition = '(-) ' + condition

                        # add in the solvent conditions
                        if sliced_df.at[row, 'Solvent NaCl Concentration'] != 0:
                            condition += ' + ' + str(sliced_df.at[row, 'Solvent NaCl Concentration']) + ' mM NaCl'
                        elif sliced_df.at[row, 'Solvent Cysteine Concentration'] != 0:
                            condition += ' + ' + str(sliced_df.at[row, 'Solvent Cysteine Concentration']) + ' mM cys'

                        # append the condition to accessed conditions and then analyze if any conditions are missing
                        accessed_conditions.append(condition)

                    # find missing conditions and insert data according to missing conditions
                    missing_conditions = set(conditions).symmetric_difference(set(accessed_conditions))

                    # loop through missing conditions
                    for condition in missing_conditions:
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
                        last_row_index = len(incomplete_a_num_df.index)

                        # insert data based on missing conditions
                        incomplete_a_num_df.at[last_row_index, 'Particle Charge'] = particle_charge
                        incomplete_a_num_df.at[last_row_index, 'Particle Size'] = particle_size

                        # inserting solvent + concentration
                        incomplete_a_num_df.at[last_row_index, 'Solvent NaCl Concentration'] = nacl_concentration
                        incomplete_a_num_df.at[last_row_index, 'Solvent Cysteine Concentration'] = cys_concentration
                        incomplete_a_num_df.at[last_row_index, 'Accession Number'] = accession_number

                        for feature in dataset.columns:
                            # ignore certain features if they've been inserted or are simply unneeded
                            if feature == 'Particle Charge' or feature == 'Particle Size' or \
                                    feature == 'Solvent NaCl Concentration' or feature == 'Solvent Cysteine Concentration' or \
                                    feature == 'Accession Number' or feature == 'Enrichment':
                                pass
                            # requires passing in a special condition string to access Bound Fraction in the InputData
                            elif feature == 'Bound Fraction':
                                incomplete_a_num_df.at[last_row_index, feature] = data.at[accession_number, condition]
                            else:
                                incomplete_a_num_df.at[last_row_index, feature] = data.at[accession_number, feature]
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


def insert_protein_abundance_data(dataset, pa_data):
    spectral_count_columns = ['C1', 'C2', 'C3']
    summed_ratios = 0
    missing_keys = set()
    dataset.insert(loc=13, column='Protein Abundance', value=np.nan)
    pa_data.insert(loc=1, column='Accession Number', value=np.nan)
    pa_data['Accession Number'] = pa_data['Accession Number'].astype('string')

    # iterate through rows of pa_data
    for row in pa_data.index:
        nan_count = 0
        # iterate through the spectral count columns C1 - C3
        for col in spectral_count_columns:
            if pd.isna(pa_data.at[row, col]):
                nan_count += 1

        # if nan_count reaches 2 or more, drop the row
        if nan_count >= 2:
            pa_data.drop(index=row, inplace=True)
        else:
            start_idx = pa_data.at[row, 'Protein Name'].find('|') + 1
            accession_number = (pa_data.at[row, 'Protein Name'])[start_idx: start_idx + 6]
            pa_data.at[row, 'Accession Number'] = accession_number

        # calculate specific (PSM/Length) ratios for each Accession Number if it's in the InputData.csv DataFrame
        try:
            length = dataset.at[accession_number, 'Protein Length']
            averaged_psm = pa_data.at[row, 'SUM'] / (3 - nan_count)
            original_data.at[accession_number, 'Protein Abundance'] = averaged_psm/length
            summed_ratios += averaged_psm/length
        except KeyError:
            missing_keys.add(accession_number)

    # to satisfy the NSAF formula, divide by summed (PSM/length) values
    dataset['Protein Abundance'] = dataset['Protein Abundance'] / summed_ratios

    return dataset


if __name__ == '__main__':
    original_data = clean_original_data()
    original_data = insert_protein_abundance_data(original_data, pd.read_csv('Input_Files/Protein_Abundance_Data.csv'))
    print(original_data)
    # print(original_data.shape)
    # print(original_data.loc['P00359'])
    # find_bound_fraction_filler(original_data)
    insert_accession_numbers(original_data)


