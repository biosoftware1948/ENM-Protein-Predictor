"""
This Module deals with data fetching and cleaning
"""
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing, cross_validation
import random

def classify(data, cutoff):
    """This function classifies particles as bound or unbound
    Takes unclassified data the cutoff as arguments
    returns classified data in an array
    """
    classified_data = np.empty((len(data)))
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    for i, val in enumerate(data):
        if val >= cutoff:
            classified_data[i] = 1
        else:
            classified_data[i] = 0

    return classified_data

def fill_nan(data, column):
    """ Fills nan values in specified column. Takes
    dataframe and column as input, returns dataframe with
    nans filled in specified column
    """
    count = 0
    total = 0
    for val in data[column]:
        if not np.isnan(val):
            count+=1
            total+=val
    data[column] = data[column].fillna(total/count)
    return data

def get_dummies(dataframe, category):
    """This function converts categorical variables into dummy variables
    Takes pandas dataframe and the catefory name as arguments
    Returns the dataframe with new dummy variables
    """
    dummy = pd.get_dummies(dataframe[category], prefix=category)
    dataframe = pd.concat([dataframe,dummy], axis = 1)
    dataframe.drop(category, axis=1, inplace=True)
    return dataframe

def clean_print(obj):
    """
    Prints the JSON in a clean format for all my
    Biochemistry friends
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

def fetch_data(enm_database):
    """
    Pulls the Data from CSV format. Returns 3012 measured protein-particle
    interactions represented as vectors
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    enm_database = os.path.join(dir_path, enm_database)

    try:
        data = pd.read_csv(enm_database)
        enrichment = np.array(data['Enrichment'])

        data = data.drop(['Enrichment'], 1)
    except:
        print "Error Fetching CSV Data"

    #One hot encoding of categorical data
    data = get_dummies(data, 'size')
    data = get_dummies(data, 'charge')
    data = get_dummies(data, 'salt')
    data = get_dummies(data, 'cysteine')
    #Clean nan values
    data = fill_nan(data, 'Abundance')
    #Normalize the data
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    df_normalized = pd.DataFrame(np_scaled)
    #Classify enrichment data, using enrichment ratio of 1
    classed_enrich = classify(enrichment, 0.5)
    #split data into training and testing set. Use testing set to validate model at the end
    X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(df_normalized, classed_enrich, test_size=0.1, random_state = random.randint(1, 2**8))
    #Ravel those vectors
    Y_test = np.ravel(Y_test)
    Y_train = np.ravel(Y_train)


    return X_train, X_test, Y_train, Y_test, enrichment, data

fetch_data("train.csv")
