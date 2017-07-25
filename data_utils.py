"""
This Module deals with data fetching and cleaning.
It includes several functions:

1) classify: classifies continous data
2) fill_nan: this fills nan values in a column with the meaningless
3) one_hot_encode: converts categorical data to one hot vectors
4) clean_print: Prints data in a clean format
5) fetch_data: Specific for our dataset, pre-processes ENM data
    and splits it accordingly
"""
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing, cross_validation
import random

def classify(data, cutoff):
    """
    This function classifies continous data.
    In our case we classify particles as bound or unbound

    Args:
        :param data (array): array of continous data
        :param cutoff (float): cutoff value for classification

    Returns:
        :classified_data(np.array): classified data
    """
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except:
            print "data could not be converted to type: numpy array"

    classified_data = np.empty((len(data)))

    for i, val in enumerate(data):
        if val >= cutoff:
            classified_data[i] = 1
        else:
            classified_data[i] = 0

    return classified_data

def fill_nan(data, column):
    """ Fills nan values with mean in specified column.

    Args:
        :param data (pandas Dataframe): Dataframe containing column with nan values
        :param column (String): specifying column to fill_nans

    Returns:
        :data (pandas Dataframe): Containing the column with filled nan values
    """
    assert isinstance(data, pd.DataFrame), 'data argument needs to be pandas dataframe'
    count = 0
    total = 0
    for val in data[column]:
        if not np.isnan(val):
            count+=1
            total+=val
    data[column] = data[column].fillna(total/count)
    return data

def one_hot_encode(dataframe, category):
    """This function converts categorical variables into one hot vectors

    Args:
        :param dataframe (pandas Dataframe): Dataframe containing column to be encoded
        :param category (String): specifying the column to encode

    Returns:
        :dataframe (Pandas Dataframe): With the specified column now encoded into a one
        hot representation
    """
    assert isinstance(dataframe, pd.DataFrame), 'data argument needs to be pandas dataframe'
    dummy = pd.get_dummies(dataframe[category], prefix=category)
    dataframe = pd.concat([dataframe,dummy], axis = 1)
    dataframe.drop(category, axis=1, inplace=True)
    return dataframe

def clean_print(obj):
    """
    Prints the JSON in a clean format for all my
    Biochemistry friends

    Args:
        :param obj (object): Any object you wish to print in readable format

    Returns:
        None
    """
    if isinstance(obj, dict):
        for key, val in obj.items():
            if hasattr(val, '__iter__'):
                print "\n" + key
                clean_print(val)
            else:
                print '%s : %s' % (key, val)
    elif isinstance(obj, list):
        for val in obj:
            if hasattr(val, '__iter__'):
                clean_print(val)
            else:
                print val
    else:
        if isinstance(obj, pd.DataFrame):
            clean_print(obj.to_dict(orient='records'))
        else:
            print str(obj) + "\n"

def fetch_data(enm_database, test_size=0.0):
    """
    Pulls the Data from CSV format. Returns 3012 measured protein-particle
    interactions represented as vectors.

    Args:
        :param enm_database (String): specifying the path of the csv file
        :param test_size (float): The percentage of the data to be used for
        testing, must be between [0, 1]

    Returns:
        :X_train (np.array floats): X values of training data
        :X_test (np.array floats): X values of testing data
        :Y_train (np.array floats): Y values of training data
        :Y_test (np.array floats): Y values of testing data
        :enrichment (np.array floats): Enrichment values
        :data (pandas dataframe): Processed database
    """
    assert isinstance(enm_database, str), "please pass a string specifying database location"
    assert test_size >= 0.0 and test_size < 1.0, "test size must be between zero and one"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    enm_database = os.path.join(dir_path, enm_database)

    try:
        data = pd.read_csv(enm_database)
        enrichment = np.array(data['Enrichment'])

        data = data.drop(['Enrichment'], 1)
    except:
        print "Error Fetching CSV Data"

    #One hot encoding of categorical data
    data = one_hot_encode(data, 'size')
    data = one_hot_encode(data, 'charge')
    data = one_hot_encode(data, 'salt')
    data = one_hot_encode(data, 'cysteine')
    #Clean nan values
    data = fill_nan(data, 'Abundance')
    #Normalize the data
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    df_normalized = pd.DataFrame(np_scaled)
    #Classify enrichment data, using enrichment ratio of 1
    classed_enrich = classify(enrichment, 0.5)
    #split data into training and testing set. Use testing set to validate model at the end
    X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(df_normalized, classed_enrich, test_size=test_size, random_state = random.randint(1, 2**8))
    #Ravel those vectors
    Y_test = np.ravel(Y_test)
    Y_train = np.ravel(Y_train)


    return X_train, X_test, Y_train, Y_test, enrichment, data
