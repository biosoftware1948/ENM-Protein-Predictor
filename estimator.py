"""
Matthew Findlay
Santa Clara University
Dr. Wheeler's Lab
Undergraduate student dept Bioengineering
2017

Modified by Joseph Pham Nguyen
Santa Clara University
Dr. Wheeler's Lab
Undergraduate Student, Dept. Computer Science and Engineering
2021

This Script predicts the amount of proteins (Bound Fraction) that are bound to the protein corona on the surface of
engineered nanomaterials. To achieve this we first experimentally isolate proteins that bind and do not bind to
engineered nanomaterials under a variety of relevant biological conditions. We send these protein samples to Stanford's
LC-MS/MS facilities to identify the proteins and their associated spectral counts. We then mine online databases to
create a database containing information about the proteins, particles, and solvent conditions.
To make predictions from our database we use a random forest regression algorithm.
We validate our regressions with several statistical methods like root mean squared error.
"""

import data_utils
import validation_utils
import visualization_utils
import numpy as np
import predictor_utils
from sklearn.ensemble import RandomForestRegressor
import sys
import json
import argparse
import os


def pipeline(db, validation, optimize, RFECV):
    """
    Runs the pipeline. Trains and evaluates the estimator, outputs metrics and
    information about the model performance.

    Args:
        :param: db (database obj): The database object, passed from main.
        Information about this class can be found in data_utils
        :param: validation (validation_utils object: this object holds onto error metrics for the
        RandomForestRegressor, and the final result will output several statistics about the model
        :param: optimize (bool): Set to true to run Grid search
        :param: RFECV (bool): Set to true to run RFECV
    Returns: None
    """
    if db.predict is None:
        # We split our own data for training and testing if user isn't predicting their own data
        db.data_split()

    # apply the RFECV mask to only keep selected features from the RFECV algorithm
    db.X_train, db.X_test = data_utils.apply_RFECV_mask('Input_Files/_mask.txt', db.X_train, db.X_test)

    est = RandomForestRegressor(
        n_estimators=2500,
        bootstrap=True,
        min_samples_leaf=1,
        min_samples_split=3,
        n_jobs=-1,
        random_state=data_utils.random.randint(1, 2 ** 8)
    )

    if optimize:
        predictor_utils.optimize(est, db.X_train, db.Y_train)
        sys.exit(0)
    if RFECV:
        predictor_utils.recursive_feature_elimination(est, db.X_train, db.Y_train, 'Input_Files/_mask.txt')
        sys.exit(0)

    est.fit(db.X_train, db.Y_train)

    # will be used to filter predicted values by protein particle properties
    conditions = db.X_test[['Particle Size_10', 'Particle Size_100',
                            'Particle Charge_0', 'Particle Charge_1', 'Solvent Cysteine Concentration_0.1',
                            'Solvent NaCl Concentration_0.8', 'Solvent NaCl Concentration_3.0']]

    # Calculate individual metrics and hold onto it until the end to calculate model information
    # Note that this eats up memory in return for convenience of outputting/formatting model information
    validation.set_parameters(db.Y_test, est.predict(db.X_test), db.test_accession_numbers, conditions)
    validation.calculate_error_metrics(), validation.update_predictions()
    validation.update_feature_importances(list(db.X_train.columns), est.feature_importances_)


class NpEncoder(json.JSONEncoder):
    """Handles simple conversions from NumPy data types to Python data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def check_file_path(file_path: str) -> str:
    if os.path.exists(file_path):
        return file_path
    else:
        print("\nInvalid file path. Please input an existing filepath to desired file.\n")


if __name__ == '__main__':
    # Initialize database
    db = data_utils.data_base()
    db.raw_data = "Input_Files/database.csv"
    db.clean_raw_data()

    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('script', help="script with which to run the model", type=str)
    parser.add_argument('iterations', help="number of times to run the model", type=int)
    parser.add_argument('-i', '--input', help="file path to your csv file to make predictions on", type=check_file_path,
                        default=None)
    parser.add_argument('-o', '--optimize', help="specify boolean value on whether or not to run GridSearchCV",
                        type=bool, default=False)
    parser.add_argument('-r', '--rfecv', help="specify boolean value on whether or not to run RFECV",
                        type=bool, default=False)
    args = vars(parser.parse_args(sys.argv))

    # process arguments
    iterations = args['iterations']
    opt = args['optimize']
    rfecv = args['rfecv']

    if args['input'] is not None:
        db.predict = args['input']

    # error metric values will be set during the pipeline
    val = validation_utils.validation_metrics()

    # Run the model multiple times and store results
    for i in range(0, iterations):
        print("Run Number: {}".format(i))
        pipeline(db, validation=val, optimize=opt, RFECV=rfecv)

    # calculate the average error metric scores + average predicted value
    average_error_metrics, predicted_values_stats, average_feature_importances, stddev = val.calculate_final_metrics()

    # visualize the feature_importances in a bar graph
    visualization_utils.visualize_feature_importances(average_feature_importances, stddev)

    # save error metrics + feature importances
    data_utils.save_metrics(average_error_metrics, average_feature_importances, predicted_values_stats)
    print("\nThe model has finished, please see the saved output files for more information and statistics\n")
