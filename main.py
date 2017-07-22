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
import visualization_tools
import predictor
import validator
import sys
import json

def main():
    X_train, X_test, Y_train, Y_test, enrichment, data = data_utils.fetch_data('train.csv', 0.1)

    est = predictor.RandomForestClassifierWithCoef(
                                 #criterion='mse',             #mean squared error criterion
                                 n_estimators=1000,             #number of trees used by the algorithm
                                 oob_score=True,               #Out of box score
                                 max_features='auto',          #features at each split (auto=all)
                                 max_depth=None,               #max tree depth
                                 min_samples_split=5,          #minimum amount of samples to split a node
                                 min_samples_leaf=1,           #minimum amount of samples a leaf can contain
                                 min_weight_fraction_leaf=0,   #minimum weight fraction of samples in a leaf
                                 max_leaf_nodes=None,          #maximum amount of leaf nodes
                                 n_jobs=-1,                    #CPU Cores used (-1 uses all)
                                 random_state=data_utils.random.randint(1, 2**8)  #Initialize random seed generator
                                 )
    est.fit(X_train, Y_train)
    probability_prediction = est.predict_proba(X_test)[:,1]
    val = validator.validation_metrics(Y_test, probability_prediction)
    val.youden_index()
    val.roc_curve()

    return val.well_rounded_validation(), dict(zip(list(data), est.feature_importances_))

if __name__ == '__main__':
    results = {}
    for i in range(0, int(sys.argv[1])):
        metrics = main()
        results["Run_" + str(i)] = metrics[0], metrics[1]
    print json.dumps(results)
