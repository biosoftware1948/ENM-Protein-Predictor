"""
Matthew Findlay
Santa Clara University
Dr. Wheeler's Lab
Undergraduate student dept Bioengineering
2017

This Script predicts if proteins will be found in the protein corona on the surface of Engineered Nanomaterials.
To achieve this we first experimentally isolate proteins that bind and do not bind to engineered nanomaterials
under a variety of relevant biological conditions. We send these protein samples to Stanford's to LC-MS/MS facilities
to identify the proteins and their associated spectral counts. We then mine online databases to create a database
containing information about the proteins, particles, and solvent conditions.
To make predictions from our database we use a random forest classification algorithm.
We validate our classifications with several statistical methods including ROC curves.
"""
import data_utils
import numpy as np
import visualization_utils
import predictor_utils
import validation_utils
import sys
import json
import os
from sklearn import metrics


def pipeline(db, validation, test_percentage=0.1, optimize=False, RFECV=False):
    """
    Runs the pipeline. Trains and evaluates the estimator, outputs metrics and
    information about the model performance.

    Args:
        :param: db (database obj): The database object, passed from main.
        Information about this class can be found in data_utils
        :param: optimize (bool): Set to true to run Grid search
        :param: RFECV (bool): Set to true to run RFECV
    Returns:
        :val.well_rounded_validation() (dict): returns a dictionary of validation metrics
        :feature_importances (dict): contains a dictionary of feature importances
        :classification_information (dict): information about the predictions
    """
    if db.predict is None:
        # We split our own data for training and testing if user isn't predicting their own data
        db.stratified_data_split()

    # apply the RFECV mask to only keep selected features from the RFECV algorithm
    db.X_train, db.X_test = data_utils.apply_RFECV_mask('Input_Files/_new_mask.txt', db.X_train, db.X_test)

    est = predictor_utils.RandomForestRegressor(
        n_estimators=2500,
        bootstrap=True,
        min_samples_split=3,
        n_jobs=-1,
        random_state=data_utils.random.randint(1, 2 ** 8)
    )

    if optimize:
        predictor_utils.optimize(est, db.X_train, db.Y_train)
        sys.exit(0)
    if RFECV:
        predictor_utils.recursive_feature_elimination(est, db.X_train, db.Y_train, 'Input_Files/_new_mask.txt')
        sys.exit(0)

    est.fit(db.X_train, db.Y_train)

    validation.set_parameters(db.Y_test, est.predict(db.X_test))
    validation.calculate_error_metrics()

    # validator.y_randomization_test(est, db)  # run y_randomization_test
    feature_importances = dict(zip(list(db.X_train), est.feature_importances_))

    return feature_importances


class NpEncoder(json.JSONEncoder):
    """Because json doesn't recognize NumPy data types, this handles simple conversions
    from NumPy data types to Python data types that json can recognize and therefore serialize"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


if __name__ == '__main__':
    assert len(sys.argv) == 3, "First command line argument is the amount of times to run the model, " \
                               "second command line argument is output file for json results"
    iterations = int(sys.argv[1])
    output_file = sys.argv[2]

    # Initialize our database
    db = data_utils.data_base()
    # This is the newly reformatted database that is being tested right now
    db.raw_data = "Reformatted_Files/_updated_database.csv"
    db.clean_raw_data()

    # To use our data to predict yours, set your data below and uncomment:
    # db.predict = "your_csv_path" #<-----Set your own data here

    # error metric values will be set during the pipeline
    val = validation_utils.validation_metrics()

    # Run the model multiple times and store results
    for i in range(0, iterations):
        print("Run Number: {}".format(i))
        metrics = pipeline(db, validation=val)

    # calculate the average of each error metric score
    average_error_metrics = val.average_error_metrics()

    # insert future code for outputting more information
