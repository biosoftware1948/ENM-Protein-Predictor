"""Developed by: Matthew Findlay 2017

This Module contains the database class that handles all of the data gathering
and cleaning. It also contains functions that help us work with our data.
"""
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing, model_selection
import random
import csv


def apply_RFECV_mask(mask, *args):
    """Applies a binary mask to a dataframe to remove columns. Binary mask is
    created from recursive feature elimination and cross validation and
    optimizes the generalization of the model

    Args:
        :param mask (string): text file containing the binary mask
        :param *args (pandas dataframe): Dataframes containing columns to mask
    Returns:
        :new dataframes (pandas df): new dataframes with columns removed
    """
    assert os.path.isfile(mask), "please pass a string specifying mask location"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    mask = os.path.join(dir_path, mask)
    # get mask data
    updated_args = []
    with open(mask, 'r') as f:
        reader = csv.reader(f)
        column_mask = list(reader)[0]
    # print("Length of column mask:" + str(len(column_mask)))
    # print("This is the column mask as a list: \n" + str(column_mask) + "\n")
    # apply mask to columns
    column_indexes = []
    for dataframe in args:
        if len(column_mask) != len(list(dataframe)):
            column_mask = remove_extra_entries(column_mask)
            assert len(column_mask) == len(list(dataframe)), 'mask length {} does not match dataframe length {}'\
                .format(len(column_mask), len(list(dataframe)))

        for i, col in enumerate(column_mask):
            if col.strip() == 'False':
                column_indexes.append(i)

        updated_args.append(dataframe.drop(dataframe.columns[column_indexes], axis=1))
    return updated_args


def remove_extra_entries(mask):
    """Remove extra entries like '' or '\n' in the binary mask iff the length of mask does not match the corresponding
    dataframe length

    Args:
        :param: mask (array): the binary mask as a list of values
    Returns: mask (array): the binary mask with extraneous values removed from end of list
    """
    for i, col in reversed(list(enumerate(mask))):
        stripped_col = col.strip()
        if stripped_col == "True" or stripped_col == "False":
            break
        else:
            del mask[i]
    return mask


class data_base(object):
    """Handles all data fetching and preparation. Attributes
       can be assigned to csv files with the assignment operator. Typical use
       case is to set raw_data to a csv file matching the format found in
       Input files and then calling clean_raw_data(). This sets the clean_X_data,
       y_enrichment and target values. From this point you can split the data
       to train/test the model using our data. To predict your own data, make sure your excel sheet
       matches the format in <Input_Files/database.csv>. Then you can
       call db.predict = <your_csv_path>. The X_test and Y_test data will now
       be your data. Just remove the stratified_data_split from the pipeline
       because you will now not need to split any data.

       Args:
            None
       Attributes:
            :self._raw_data (Pandas Dataframe): Holds raw data in the same form as excel file. initialized after fetch_raw_data() is called
            :self._clean_X_data (Pandas Dataframe): Holds cleaned and prepared X data.
            :self._target (np.array): holds target values for predictions
            :self._Y_enrichment (numpy array): Holds continous Y values # REMOVED/COMMENTED
            :self._X_train (Pandas Dataframe): Holds the X training data
            :self._X_test (Pandas Dataframe): Holds the X test data
            :self._Y_train (Pandas Dataframe): Holds the Y training data
            :self._Y_test (Pandas Dataframe): Holds the T testing data
            :self._test_accession_numbers (list): holds the accession_numbers
            in the test set
        """
    categorical_data = ['Enzyme Commission Number', 'Particle Size', 'Particle Charge', 'Solvent Cysteine Concentration', 'Solvent NaCl Concentration']
    columns_to_drop = ['Protein Length', 'Sequence', 'Accession Number', 'Bound Fraction']

    def __init__(self):
        self._raw_data = None
        self._clean_X_data = None
        # self._Y_enrichment = None
        self._target = None
        self._X_train = None
        self._Y_train = None
        self._X_test = None
        self._Y_test = None
        self._test_accession_numbers = None
        self._original = None
        # If you want to use our model set this to your csv file using the assignment operator
        self._predict = None

    def clean_raw_data(self):
        """ Cleans the raw data, drops useless columns, one hot encodes, and extracts
        class information

        Args, Returns: None
        """
        self.clean_X_data = self.raw_data
        # one hot encode categorical data
        for category in self.categorical_data:
            self.clean_X_data = one_hot_encode(self.clean_X_data, category)

        # Fill in missing values in target label and 'Protein Abundance before dropping independent variables
        self.clean_X_data['Bound Fraction'].fillna(self.clean_X_data['Bound Fraction'].mean())

        # grab target label and accession numbers
        self._target = self.clean_X_data['Bound Fraction'].to_numpy()
        self.Y_train = self.target
        accession_numbers = self.clean_X_data['Accession Number']

        # drop useless columns
        for column in self.columns_to_drop:
            self.clean_X_data = self.clean_X_data.drop(column, 1)

        # TEST: fill with either the mean average or with zeroes
        self.clean_X_data = fill_nan(self.clean_X_data, 'Protein Abundance')
        # self.clean_X_data = fill_zero(self.clean_X_data, 'Protein Abundance')

        # This grabs the original cleaned data so that it can be visualized in visualization_utils.py
        self._original = self.clean_X_data
        self.clean_X_data = normalize_and_reshape(self.clean_X_data, accession_numbers)
        self.X_train = self.clean_X_data

    def clean_user_test_data(self, user_data):
        """This method makes it easy for other people to make predictions
        on their data.
        called by assignment operator when users set db.predict = <path_to_csv>

        Args:
            :param user_data: users data they wish to predict
        Returns:
            None
        """
        # one hot encode categorical data
        for category in self.categorical_data:
            user_data = one_hot_encode(user_data, category)

        # Grab some useful data before dropping from independent variables
        self.Y_test = user_data['Bound Fraction'].to_numpy()
        accession_numbers = user_data['Accession Number']

        for column in self.columns_to_drop:
            user_data = user_data.drop(column, 1)

        user_data = fill_nan(user_data, 'Protein Abundance')
        self.X_test = normalize_and_reshape(user_data, accession_numbers)

        # Get accession number
        self.test_accession_numbers = self.X_test['Accession Number']
        self.X_train = self.X_train.drop('Accession Number', 1)
        self.X_test = self.X_test.drop('Accession Number', 1)

    def stratified_data_split(self):
        """Uses KFold Cross Validation with 5 folds to randomly split our data into training and testing sets
        Args, Returns: None
        """
        assert self.predict is None, "Remove stratified_data_split() if using your own data"

        # Testing randomized shuffle versus non-randomized splitting for data training and testing sets
        # kf = model_selection.KFold(n_splits=5, random_state=int((random.random()*100)), shuffle=True)
        kf = model_selection.KFold(n_splits=5, random_state=42, shuffle=True)
        # kf = model_selection.KFold(n_splits=5)
        for train_index, test_index in kf.split(self.clean_X_data):
            self.X_train, self.X_test = self.clean_X_data.iloc[list(train_index)], self.clean_X_data.iloc[list(test_index)]
            self.Y_train, self.Y_test = self.target[train_index], self.target[test_index]

        self.test_accession_numbers = self.X_test['Accession Number']
        self.X_train = self.X_train.drop('Accession Number', 1)
        self.X_test = self.X_test.drop('Accession Number', 1)

    @staticmethod
    def fetch_raw_data(enm_database):
        """Fetches enm-protein data from a csv file
        called by assignment operator for db.raw_data

        Args:
            :param enm_database (str): path to csv database
        Returns:
            None
        """
        assert os.path.isfile(enm_database), "please pass a string specifying database location"

        dir_path = os.path.dirname(os.path.realpath(__file__))
        enm_database = os.path.join(dir_path, enm_database)
        try:
            raw_data = pd.read_csv(enm_database)
        except ValueError:
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
    def original(self):
        if self._original is None:
            raise ValueError("Initialize original by calling clean_data()")
        else:
            return self._original

    @property
    def target(self):
        if self._target is None:
            raise ValueError("Initialize target by calling clean_data()")
        else:
            return self._target

    @property
    def test_accession_numbers(self):
        if self._test_accession_numbers is None:
            raise ValueError("Initialize test_accession_numbers by calling stratified_data_split()")
        else:
            return self._test_accession_numbers

    @property
    def predict(self):
        return self._predict

    @X_train.setter
    def X_train(self, path):
        if isinstance(path, str) and os.path.isfile(path):
            # If trying to set to value from excel
            self._X_train = self.fetch_raw_data(path)
        else:
            # If trying to set to already imported array
            self._X_train = path

    @X_test.setter
    def X_test(self, path):
        if isinstance(path, str) and os.path.isfile(path):
            # If trying to set to value from excel
            self._X_test = self.fetch_raw_data(path)
        else:
            # If trying to set to already imported array
            self._X_test = path

    @Y_train.setter
    def Y_train(self, path):
        if isinstance(path, str) and os.path.isfile(path):
            # If trying to set to value from excel
            self._Y_train = self.fetch_raw_data(path)
        else:
            # If trying to set to already imported array
            self._Y_train = path

    @Y_test.setter
    def Y_test(self, path):
        if isinstance(path, str) and os.path.isfile(path):
            # If trying to set to value from excel
            self._Y_test = self.fetch_raw_data(path)
        else:
            # If trying to set to already imported array
            self._Y_test = path

    @raw_data.setter
    def raw_data(self, enm_database):
        if isinstance(enm_database, str) and os.path.isfile(enm_database):
            self._raw_data = self.fetch_raw_data(enm_database)
        else:
            self._raw_data = enm_database

    @clean_X_data.setter
    def clean_X_data(self, path):
        if isinstance(path, str) and os.path.isfile(path):
            # If trying to set to value from excel
            self.clean_X_data = self.fetch_raw_data(path)
        else:
            # If trying to set to already imported array
            self._clean_X_data = path

    @test_accession_numbers.setter
    def test_accession_numbers(self, path):
        if isinstance(path, str) and os.path.isfile(path):
            # If trying to set to value from excel
            # self._Y_enrichment = fetch_raw_data(path)
            print("Enrichment factors don't exist anymore")
        else:
            # If trying to set to already imported array
            self._test_accession_numbers = path

    @predict.setter
    def predict(self, path):
        if os.path.isfile(path):
            self._predict = self.fetch_raw_data(path)
            self._predict = self.clean_user_test_data(self._predict)
        else:
            self._predict = path


def normalize_and_reshape(data, labels):
    """Normalize and reshape the data by columns while preserving labels
    information

    Args:
        :param data (pandas df): The data to normalize
        :param labels (pandas series): The column labels
    Returns:
        :param data (pandas df): normalized dataframe with preserved column labels
    """
    norm_df = preprocessing.MinMaxScaler().fit_transform(data)
    data = pd.DataFrame(norm_df,columns=list(data))
    data = pd.concat([labels, data], axis=1)
    data.reset_index(drop=True, inplace=True)
    return data


def fill_nan(data, column):
    """ Fills nan values with mean in specified column.

    Args:
        :param: data (pandas Dataframe): Dataframe containing column with nan values
        :param: column (String): specifying column to fill_nans
    Returns:
        :data (pandas Dataframe): Containing the column with filled nan values
    """
    assert isinstance(data, pd.DataFrame), 'data argument needs to be pandas dataframe'
    assert isinstance(column, str), 'Column must be a string'

    count = 0
    total = 0
    for val in data[column]:
        if not np.isnan(val):
            count += 1
            total += val
    data[column] = data[column].fillna(total/count)
    return data


def fill_zero(data, column):
    """Fills nan values with 0's in the specified column

    Args:
        :param: data (pandas Dataframe): Dataframe containing column with nan values
        :param: column (String): specifying column to fill_nans
    Returns:
        :data (pandas Dataframe): Containing the column with filled nan values
    """
    assert isinstance(data, pd.DataFrame), 'data argument needs to be pandas dataframe'
    assert isinstance(column, str), 'Column must be a string'
    data[column] = data[column].fillna(0)
    print("Printing values from specified column to check, should contain 0's in place of NaNs.\n")
    print(list(data[column].values))
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
    dataframe = pd.concat([dataframe, dummy], axis=1)
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
                print("\n" + key)
                clean_print(val)
            else:
                print('%s : %s' % (key, val))
    elif isinstance(obj, list):
        for val in obj:
            if hasattr(val, '__iter__'):
                clean_print(val)
            else:
                print(val)
    else:
        if isinstance(obj, pd.DataFrame):
            clean_print(obj.to_dict(orient='records'))
        else:
            print(str(obj) + "\n")


def to_excel(classification_information):
    """ Prints model output to an excel file

        Args:
            :classification_information (numpy array): Information about results
            >classification_information = {
                'all_predict_proba' : np.empty([TOTAL_TESTED_PROTEINS], dtype=float),
                'all_true_results' : np.empty([TOTAL_TESTED_PROTEINS], dtype=int),
                'all_accession_numbers' : np.empty([TOTAL_TESTED_PROTEINS], dtype=str),
                'all_particle_information' : np.empty([2, TOTAL_TESTED_PROTEINS], dtype=int),
                'all_solvent_information' : np.empty([3, TOTAL_TESTED_PROTEINS], dtype=int)
                }
        Returns:
            None
        """
    with open('prediction_probability.csv', 'w') as file:
        file.write('Protein Accession Number, Particle Type, Solvent Conditions, True Bound Value, Predicted Bound Value, Predicted Probability of Being Bound, Properly Classified\n')

        for pred, true_val, protein, particle_s, particle_c, cys, salt8, salt3, in zip(classification_information['all_predict_proba'],
                                                                                       classification_information['all_true_results'],
                                                                                       classification_information['all_accession_numbers'],
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
            if particle_size == '10nm' and particle_charge == 'positive':
                particle = '(+) 10 nm AgNP'
            if particle_size == '10nm' and particle_charge == 'negative':
                particle = '(-) 10 nm AgNP'
            if particle_size == '100nm' and particle_charge == 'negative':
                particle = '(-) 100 nm AgNP'
            if cys == 1:
                solvent = '10 mM NaPi pH 7.4 + 0.1 mM cys'
            if salt8 == 1:
                solvent = '10 mM NaPi pH 7.4 + 0.8 mM NaCl'
            if salt3 == 1:
                solvent = '10 mM NaPi pH 7.4 + 3.0 mM NaCl'

            file.write('{}, {}, {}, {}, {}, {}, {}\n'.format(protein, particle, solvent, bound, predicted_bound,round(pred, 2), properly_classified))


def hold_in_memory(classification_information, metrics, iterations, test_size):
    """Holds classification data in memory to be exported to excel

    Args:
        :classification_information (dict): container for all the classification_information from all the runs
        :metrics (tuple): information from the current test set to add to classification_information
        :iterations (int): The current test iterations
        :test_size (int): The amount of values in the current test set
    Returns:
        None
    """
    i = iterations
    TEST_SIZE = test_size # 10% of training data is used for testing ceil(10% of 3012)=302
    PARTICLE_SIZE = 0
    PARTICLE_CHARGE = 1
    SOLVENT_CYS = 0
    SOLVENT_SALT_08 = 1
    SOLVENT_SALT_3 = 2
    # Information is placed into numpy arrays as blocks
    classification_information['all_predict_proba'][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[0]
    classification_information['all_true_results'][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[1]
    classification_information['all_accession_numbers'][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[2]
    classification_information['all_particle_information'][PARTICLE_CHARGE][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[3]['Particle Charge_1']
    classification_information['all_particle_information'][PARTICLE_SIZE][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[3]['Particle Size_10']
    classification_information['all_solvent_information'][SOLVENT_CYS][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[3]['Solvent Cysteine Concentration_0.1']
    classification_information['all_solvent_information'][SOLVENT_SALT_08][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[3]['Solvent NaCl Concentration_0.8']
    classification_information['all_solvent_information'][SOLVENT_SALT_3][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[3]['Solvent NaCl Concentration_3.0']


if __name__ == "__main__":
    db = data_base()
    db.clean_data()
