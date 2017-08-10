"""
Matthew Findlay
Santa Clara University
Dr. Wheeler's Lab

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
import visualization_tools
import predictor_utils
import validator
import sys
import json
import numpy as np

def main(db):
    #Clean data
    db.split_data(0.1)
    db.X_train, db.X_test= data_utils.apply_RFECV_mask('Input_Files/_mask.txt', db.X_train, db.X_test)
    """
    Overloaded Random Forest Classifier with coefficients
    to use for recursive feature elimination and cross validation
    """
    est = predictor_utils.RandomForestClassifierWithCoef(
                                 n_estimators=1000,             #number of trees used by the algorithm
                                 bootstrap=True,
                                 oob_score=True,               #Out of box score
                                 max_features='auto',          #features at each split (auto=all)
                                 max_depth=None,               #max tree depth
                                 min_samples_split=4,          #minimum amount of samples to split a node
                                 min_samples_leaf=1,           #minimum amount of samples a leaf can contain
                                 min_weight_fraction_leaf=0,   #minimum weight fraction of samples in a leaf
                                 max_leaf_nodes=None,          #maximum amount of leaf nodes
                                 n_jobs=-1,                    #CPU Cores used (-1 uses all)
                                 random_state=data_utils.random.randint(1, 2**8)  #Initialize random seed generator
                                 )

    #predictor_utils.optimize(est, db.X_train, db.Y_train)
    #predictor_utils.recursive_feature_elimination(est, db.X_train, db.Y_train, 'tst.txt')

    est.fit(db.X_train, db.Y_train)

    probability_prediction = est.predict_proba(db.X_test)[:,1]
    #probability_prediction_train = est.predict_proba(db.X_train)[:,1]
    val = validator.validation_metrics(db.Y_test, probability_prediction)
    #val_train =validator.validation_metrics(db.Y_train, probability_prediction_train)
    #print val_train.well_rounded_validation()
    classification_information = (probability_prediction, db.Y_test, db.test_accesion_numbers, db.X_test)
    #Remove comments to visualize validation metrics
    #val.youden_index()
    #val.roc_curve()
    return val.well_rounded_validation(), dict(zip(list(db.X_train), est.feature_importances_)), classification_information

if __name__ == '__main__':
    assert len(sys.argv) == 3, "First command line argument is the amount of times to run the model, second command line argument is output file for json results"
    iterations = int(sys.argv[1])
    output_file = sys.argv[2]

    #Set constants for array indexs
    TEST_SIZE = 302 #10% of training data is used for testing 10% of 3012=302
    TOTAL_TESTED_PROTEINS = TEST_SIZE*iterations
    SCORES = 0
    IMPORTANCES = 1
    INFORMATION = 2
    PARTICLE_SIZE = 0
    PARTICLE_CHARGE = 1
    SOLVENT_CYS = 0
    SOLVENT_SALT_08 = 1
    SOLVENT_SALT_3 = 2

    results = {}
    #Information about classified particle protein pairs
    classification_information = {'all_predict_proba' : np.empty([TOTAL_TESTED_PROTEINS], dtype=float),
                                  'all_true_results' : np.empty([TOTAL_TESTED_PROTEINS], dtype=int),
                                  'all_accesion_numbers' : np.empty([TOTAL_TESTED_PROTEINS], dtype="S10"),
                                  'all_particle_information' : np.empty([2, TOTAL_TESTED_PROTEINS], dtype=int),
                                  'all_solvent_information' : np.empty([3, TOTAL_TESTED_PROTEINS], dtype=int)
                                  }

    db = data_utils.data_base()
    db.clean_data()

    for i in range(0, iterations):
        print "Run Number: {}".format(i)
        metrics = main(db)
        #hold scores and importance data in json format
        results["Run_" + str(i)] = {'scores': metrics[SCORES], 'importances': metrics[IMPORTANCES]}
        #hold classification information in arrays to output to excel file
        #Information is placed into numpy arrays as blocks for efficiency
        classification_information['all_predict_proba'][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[INFORMATION][0]
        classification_information['all_true_results'][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[INFORMATION][1]
        classification_information['all_accesion_numbers'][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[INFORMATION][2]
        classification_information['all_particle_information'][PARTICLE_CHARGE][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[INFORMATION][3]['particle_charge_1']
        classification_information['all_particle_information'][PARTICLE_SIZE][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[INFORMATION][3]['particle_size_10']
        classification_information['all_solvent_information'][SOLVENT_CYS][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[INFORMATION][3]['solvent_cys_0.1']
        classification_information['all_solvent_information'][SOLVENT_SALT_08][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[INFORMATION][3]['solvent_salt_0.8']
        classification_information['all_solvent_information'][SOLVENT_SALT_3][i*TEST_SIZE:(i*TEST_SIZE)+TEST_SIZE] = metrics[INFORMATION][3]['solvent_salt_3.0']


    #dump the statistic results as json
    with open(output_file, 'w') as f:
        json.dump(results, f)

    #Pass classification information to be inserted into excel document
    data_utils.to_excel(classification_information)
