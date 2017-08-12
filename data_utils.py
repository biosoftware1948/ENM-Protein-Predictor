"""Developed by: Matthew Findlay 2017

This Module contains the database class that handles all of the data gathering
and cleaning. It also contains functions that help us work with our data.
"""
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing, cross_validation, model_selection
import random
import csv
import sys


def apply_RFECV_mask(mask, *args):
    """Applies a binary mask to a dataframe to remove columns. Binary mask is
    created from recursive feature elimination and cross validation and
    optimizes the generalization of the model

    Args:
        :param dataframe (pandas dataframe): Dataframe containing columns
        :param mask (string): text file containing the binary mask

    Returns:
        :new dataframen (pandas df): new dataframe with columns removed
    """
    assert os.path.isfile(mask), "please pass a string specifying mask location"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mask = os.path.join(dir_path, mask)

    updated_args = []
    with open(mask, 'r') as f:
        reader = csv.reader(f)
        column_mask = list(reader)[0]

    column_indexes = []
    for dataframe in args:
        assert len(column_mask) == len(list(dataframe)), 'mask length {} does not match dataframe length {}'.format(len(column_mask), len(list(dataframe)))
        for i, col in enumerate(column_mask):
    	    if col.strip() == 'False':
    		    column_indexes.append(i)

        updated_args.append(dataframe.drop(dataframe.columns[column_indexes], axis=1))

    return updated_args



class data_base(object):
    """Handles all data fetching and preparation. Also holds all data

       Args:
            None

       Attributes:
            :self._raw_data (Pandas Dataframe): Holds raw data in the same form as excel file. initialized after fetch_raw_data() is called

            ###attributes below are initialized after clean_data() is called###

            :self._clean_X_data (Pandas Dataframe): Holds cleaned and prepared X data.
            :self._Y_enrichmet (numpy array): Holds continous Y values
            :self._X_train (Pandas Dataframe): Holds the X training data
            :self._X_test (Pandas Dataframe): Holds the X test data
            :self._Y_train (Pandas Dataframe): Holds the Y training data
            :self._Y_test (Pandas Dataframe): Holds the T testing data
            :self._column_headers (list): holds the column headers
        """

    _ENRICHMENT_SPLIT_VALUE = 1 #enrichment threshold to classify as bound or unbound
    categorical_data = ['Enzyme Commission Number', 'Particle Size', 'Particle Charge', 'Solvent Cysteine Concentration', 'Solvent NaCl Concentration']
    columns_to_drop = ['Protein Length', 'Sequence', 'Enrichment', 'Accesion Number']

    def __init__(self):
        self._raw_data = None
        self._clean_X_data = None
        self._Y_enrichment = None
        self._target = None
        self._X_train = None
        self._Y_train = None
        self._X_test = None
        self._Y_test = None
        self._test_accesion_numbers = None

    def clean_raw_data(self):
        self.clean_X_data = self.raw_data
        #Categorize Interprot identifiers n hot encoding
        self.clean_X_data = multi_label_encode(self.clean_X_data, 'Interprot')
        #one hot encode categorical data
        for category in self.categorical_data:
            self.clean_X_data = one_hot_encode(self.clean_X_data, category)

        #Grab some useful data before dropping from independant variables
        self.Y_enrichment = self.clean_X_data['Enrichment']
        accesion_numbers = self.clean_X_data['Accesion Number']

        for column in self.columns_to_drop:
            self.clean_X_data = self.clean_X_data.drop(column, 1)

        self.clean_X_data = fill_nan(self.clean_X_data, 'Protein Abundance')
        self.clean_X_data = normalize_and_reshape(self.clean_X_data, accesion_numbers)
        self._target = classify(self.Y_enrichment, self._ENRICHMENT_SPLIT_VALUE) #enrichment or nsaf

    def stratified_data_split(self, test_size=0.0):
        assert test_size <= 1.0 and test_size >= 0.0, "test_size must be between 0 and 1"
        self.X_train, self.X_test, self.Y_train, self.Y_test = model_selection.train_test_split(self.clean_X_data, self.target, test_size = test_size, random_state=int((random.random()*100)))
        self.test_accesion_numbers = self.X_test['Accesion Number']
        self.X_train = self.X_train.drop('Accesion Number', 1)
        self.X_test = self.X_test.drop('Accesion Number', 1)

    @staticmethod
    def fetch_raw_data(enm_database):
        assert os.path.isfile(enm_database), "please pass a string specifying database location"

        dir_path = os.path.dirname(os.path.realpath(__file__))
        enm_database = os.path.join(dir_path, enm_database)
        try:
            raw_data = pd.read_csv(enm_database)
        except:
            raise ValueError("File is not a valid csv")

        return raw_data

    @property
    def X_train(self):
        if self._X_train is None:
            raise ValueError("Initialize X_train by calling stratified_data_split()")
        else:
            return self._X_train

    @property
    def X_test(self):
        if self._X_test is None:
            raise ValueError("Initialize X_test by calling stratified_data_split()")
        else:
            return self._X_test

    @property
    def Y_train(self):
        if self._Y_train is None:
            raise ValueError("Initialize Y_train by calling stratified_data_split()")
        else:
            return self._Y_train

    @property
    def Y_test(self):
        if self._Y_test is None:
            raise ValueError("Initialize Y_train by calling stratified_data_split()")
        else:
            return self._Y_test

    @property
    def raw_data(self):
        if self._raw_data is None:
            raise ValueError("Initialize raw data by setting raw_data=<path.csv>")
        return self._raw_data

    @property
    def clean_X_data(self):
        if self._clean_X_data is None:
            raise ValueError("Initialize clean_X_data by calling clean_data()")
        else:
            return self._clean_X_data

    @property
    def Y_enrichment(self):
        if self._Y_enrichment is None:
            raise ValueError("Initialize Y_enrichment by calling clean_data()")
        else:
            return self._Y_enrichment

    @property
    def target(self):
        if self._target is None:
            raise ValueError("Initialize target by calling clean_data()")
        else:
            return self._target

    @property
    def test_accesion_numbers(self):
        if self._test_accesion_numbers is None:
            raise ValueError("Initialize test_accesion_numbers by calling stratified_data_spli()t")
        else:
            return self._test_accesion_numbers

    @X_train.setter
    def X_train(self, path):
        if (isinstance(path, str) and os.path.isfile(path)):
            #If trying to set to value from excel
            self._X_train = fetch_raw_data(path)
        else:
            #If trying to set to already imported array
            self._X_train = path

    @X_test.setter
    def X_test(self, path):
        if (isinstance(path, str) and os.path.isfile(path)):
            #If trying to set to value from excel
            self._X_test = fetch_raw_data(path)
        else:
            #If trying to set to already imported array
            self._X_test = path

    @Y_train.setter
    def Y_train(self, path):
        if (isinstance(path, str) and os.path.isfile(path)):
            #If trying to set to value from excel
            self._Y_train = fetch_raw_data(path)
        else:
            #If trying to set to already imported array
            self._Y_train = path

    @Y_test.setter
    def Y_test(self, path):
        if (isinstance(path, str) and os.path.isfile(path)):
            #If trying to set to value from excel
            self._Y_test = fetch_raw_data(path)
        else:
            #If trying to set to already imported array
            self._Y_test = path

    @raw_data.setter
    def raw_data(self, enm_database):
        if (isinstance(enm_database, str) and os.path.isfile(enm_database)):
            self._raw_data = self.fetch_raw_data(enm_database)
        else:
            self._raw_data = enm_database

    @clean_X_data.setter
    def clean_X_data(self, path):
        if (isinstance(path, str) and os.path.isfile(path)):
            #If trying to set to value from excel
            self.clean_X_data = fetch_raw_data(path)
        else:
            #If trying to set to already imported array
            self._clean_X_data = path

    @Y_enrichment.setter
    def Y_enrichment(self, path):
        if (isinstance(path, str) and os.path.isfile(path)):
            #If trying to set to value from excel
            self._Y_enrichment = fetch_raw_data(path)
        else:
            #If trying to set to already imported array
            self._Y_enrichment = path

    @test_accesion_numbers.setter
    def test_accesion_numbers(self, path):
        if (isinstance(path, str) and os.path.isfile(path)):
            #If trying to set to value from excel
            self._Y_enrichment = fetch_raw_data(path)
        else:
            #If trying to set to already imported array
            self._test_accesion_numbers = path

def normalize_and_reshape(data, labels):
    norm_df = preprocessing.MinMaxScaler().fit_transform(data)
    data = pd.DataFrame(norm_df,columns=list(data))
    data = pd.concat([labels, data], axis=1)
    data.reset_index(drop=True, inplace=True)
    return data

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
        if float(val) >= float(cutoff):
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
    assert isinstance(column, str), 'Column must be a string'

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


def multi_label_encode(dataframe, column):
    """This function is used as a multilabel encoder for the Interprot numbers in the database.
        The interprot numbers are seperated by a semi-colon. We use a multi label encoder because
        a protein can have several function domains. This injects a multi-hot style encoding into the database

        Args:
            :param dataframe (Pandas Dataframe): Dataframe containing protein data
            :param column: (String): Name of column to be multi-label-encoded

        Returns:
            :new_dataframe (Pandas Dataframe): With new multi label columns
    """
    dataframe.reset_index(drop=True, inplace=True)
    interprot_identifiers = []
    protein_ips = {}

    for row, iprot in enumerate(dataframe[column].values):
        ip_list = [i for i in iprot.split(';') if i != '']
        protein_ips[row] = []
        for ip in ip_list:
            interprot_identifiers.append(ip)
            protein_ips[row].append(ip)

    categorical_df = pd.DataFrame(index= np.arange(dataframe.shape[0]), columns = set(interprot_identifiers))
    categorical_df = categorical_df.fillna(0)

    for key, val in protein_ips.iteritems():
        for v in val:
            if v != 0:
                categorical_df.set_value(key, v, 1)

    dataframe = dataframe.drop(column, 1)
    new_dataframe = pd.concat([dataframe, categorical_df], axis=1)
    return new_dataframe

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

def to_excel(classification_information):
    """ Prints model output to an excel file

        Args:
            :classification_information (numpy array): Information about results

            classification_information = {'all_predict_proba' : np.empty([TOTAL_TESTED_PROTEINS], dtype=float),
                                      'all_true_results' : np.empty([TOTAL_TESTED_PROTEINS], dtype=int),
                                      'all_accesion_numbers' : np.empty([TOTAL_TESTED_PROTEINS], dtype=str),
                                      'all_particle_information' : np.empty([2, TOTAL_TESTED_PROTEINS], dtype=int),
                                      'all_solvent_information' : np.empty([3, TOTAL_TESTED_PROTEINS], dtype=int)
                                      }

        Returns:
            None
        """

    with open('prediction_probability.csv', 'w') as file:
        file.write('Protein Accesion Number, Particle Type, Solvent Conditions, True Bound Value, Predicted Bound Value, Predicted Probability of Being Bound, Properly Classified\n')

        for pred, true_val, protein, particle_s, particle_c, cys, salt8, salt3, in zip(classification_information['all_predict_proba'],
                                                                                       classification_information['all_true_results'],
                                                                                       classification_information['all_accesion_numbers'],
                                                                                       classification_information['all_particle_information'][0],
                                                                                       classification_information['all_particle_information'][1],
                                                                                       classification_information['all_solvent_information'][0],
                                                                                       classification_information['all_solvent_information'][1],
                                                                                       classification_information['all_solvent_information'][2]
                                                                                       ):
            bound = 'no'
            predicted_bound = 'no'
            properly_classified = 'no'
            particle_charge = 'negative'
            particle_size = '10nm'
            solvent = '10 mM NaPi pH 7.4'

            if int(round(pred)) == true_val:
                properly_classified = 'yes'
            if true_val == 1:
                bound = 'yes'
            if int(round(pred)) == 1:
                predicted_bound = 'yes'
            if particle_s == 0:
                particle_size = '100nm'
            if particle_c == 1:
                particle_charge = 'positive'
            if (particle_size == '10nm' and particle_charge == 'positive'):
                particle = '(+) 10 nm AgNP'
            if (particle_size == '10nm' and particle_charge == 'negative'):
                particle = '(-) 10 nm AgNP'
            if (particle_size == '100nm' and particle_charge == 'negative'):
                particle = '(-) 100 nm AgNP'
            if (cys == 1):
                solvent = '10 mM NaPi pH 7.4 + 0.1 mM cys'
            if (salt8 == 1):
                solvent = '10 mM NaPi pH 7.4 + 0.8 mM NaCl'
            if (salt3 == 1):
                solvent = '10 mM NaPi pH 7.4 + 3.0 mM NaCl'

            file.write('{}, {}, {}, {}, {}, {}, {}\n'.format(protein, particle, solvent, bound, predicted_bound,round(pred, 2), properly_classified))

def hold_in_memory(classification_information, metrics, i):
    #Set constants for array indexs
    ITERATIONS = i
    TEST_SIZE = 302 #10% of training data is used for testing 10% of 3012=302
    PARTICLE_SIZE = 0
    PARTICLE_CHARGE = 1
    SOLVENT_CYS = 0
    SOLVENT_SALT_08 = 1
    SOLVENT_SALT_3 = 2
    #Information is placed into numpy arrays as blocks
    classification_information['all_predict_proba'][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[0]
    classification_information['all_true_results'][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[1]
    classification_information['all_accesion_numbers'][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[2]
    classification_information['all_particle_information'][PARTICLE_CHARGE][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[3]['Particle Charge_1']
    classification_information['all_particle_information'][PARTICLE_SIZE][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[3]['Particle Size_10']
    classification_information['all_solvent_information'][SOLVENT_CYS][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[3]['Solvent Cysteine Concentration_0.1']
    classification_information['all_solvent_information'][SOLVENT_SALT_08][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[3]['Solvent NaCl Concentration_0.8']
    classification_information['all_solvent_information'][SOLVENT_SALT_3][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[3]['Solvent NaCl Concentration_3.0']

if __name__ == "__main__":
    db = data_base()
    db.clean_data()
