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
import numpy as np
import os
import math

def pipeline(db, test_percentage=0.1, optimize=False, RFECV=False):
    """
    Runs the pipeline. Trains and evaluates the estimator, outputs metrics and
    information about the model performance.

    Args:
        :param db (database obj): The database object, passed from main.
        Information about this class can be found in data_utils
        :param optimize (bool): Set to true to run Grid search
        :param RFECV (bool): Set to true to run RFECV
    Returns:
        :val.well_rounded_validation() (dict): returns a dictionary of validation metrics
        :feature_importances (dict): contains a dictionary of feature importances
        :classification_information (dict): information about the predictions
    """
    if (db.predict is None):
        #We split our own data for training and testing if user isn't predicting their own data
        db.stratified_data_split(test_percentage)

    db.X_train, db.X_test= data_utils.apply_RFECV_mask('Input_Files/_mask.txt', db.X_train, db.X_test)
    #overloaded RandomForestClassifier with coef
    est = predictor_utils.RandomForestClassifierWithCoef(
                            n_estimators=1000,
                            bootstrap=True,
                            min_samples_split=4,
                            n_jobs=-1,
                            random_state=data_utils.random.randint(1, 2**8)
                            )
    if optimize:
        predictor_utils.optimize(est, db.X_train, db.Y_train)
        sys.exit(0)
    if RFECV:
        predictor_utils.recursive_feature_elimination(est, db.X_train, db.Y_train, 'tst.txt')
        sys.exit(0)

    est.fit(db.X_train, db.Y_train)
    probability_prediction = est.predict_proba(db.X_test)[:,1]

    #validator.y_randomization_test(est, db) #run y_randomization_test
    val = validation_utils.validation_metrics(db.Y_test, probability_prediction)
    classification_information = (probability_prediction, db.Y_test, db.test_accesion_numbers, db.X_test)
    feature_importances = dict(zip(list(db.X_train), est.feature_importances_))
    #Remove comments to visualize roc curve and youden index
    #val.youden_index()
    #val.roc_curve()
    return val.well_rounded_validation(), feature_importances, classification_information

if __name__ == '__main__':
    assert len(sys.argv) == 3, "First command line argument is the amount of times to run the model, second command line argument is output file for json results"
    iterations = int(sys.argv[1])
    output_file = sys.argv[2]

    #Initialize our database
    db = data_utils.data_base()
    db.raw_data = "Input_Files/database.csv"
    db.clean_raw_data()

    ###To use our data to predict yours, set your data below and uncomment:
    #db.predict = "your_csv_path" #<-----Set your own data here

    #Set constants for array indexs
    if (db.Y_test is not None):
        #db.Y_test is set if user wants to predict their own data
        test_size = db.Y_test.shape[0] #If user has their own data
    else:
        #If not we split our own database for training and testing
        test_size = 302 #10% of training data is used for testing 10% of 3012=302
    TOTAL_TESTED_PROTEINS = test_size*iterations
    SCORES = 0
    IMPORTANCES = 1
    INFORMATION = 2
    results = {}

    #Information about classified particle protein pairs
    classification_information = {'all_predict_proba' : np.empty([TOTAL_TESTED_PROTEINS], dtype=float),
                                  'all_true_results' : np.empty([TOTAL_TESTED_PROTEINS], dtype=int),
                                  'all_accesion_numbers' : np.empty([TOTAL_TESTED_PROTEINS], dtype="S10"),
                                  'all_particle_information' : np.empty([2, TOTAL_TESTED_PROTEINS], dtype=int),
                                  'all_solvent_information' : np.empty([3, TOTAL_TESTED_PROTEINS], dtype=int)
                                  }

    #Run the model multiple times and store results
    for i in range(0, iterations):
        print "Run Number: {}".format(i)
        metrics = pipeline(db)
        #hold scores and importance data in json format
        results["Run_" + str(i)] = {'scores': metrics[SCORES], 'importances': metrics[IMPORTANCES]}
        #hold classification information in arrays to output to excel file
        data_utils.hold_in_memory(classification_information, metrics[INFORMATION], i, test_size)

    #dump the statistic and feature importance results as json
    with open(output_file, 'w') as f:
        json.dump(results, f)
    #Pass prediction information to be inserted into excel document
    data_utils.to_excel(classification_information)
    #Run statistic parser for human readable json
    os.system('python statistic_parser.py {}'.format(output_file))
