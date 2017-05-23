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
import math
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sys import argv
import sys
import json

class RandomForestClassifierWithCoef(RandomForestClassifier):
    """Adds feature weights for each returned variable"""
    def fit(self, *args, **kwargs):
        """Overloaded fit method to include the feature importances
        of each variable. This is used for RFECV
        """
        super(RandomForestClassifierWithCoef, self).fit(*args, **kwargs)
        self.coef_ = self.feature_importances_

class validation_metrics(object):
    """Several statistical tests used to validate the models predictive power
    Takes True results of the test data and the models results as parameters for constructor
    """
    def __init__(self, true_results, predicted_results):
        self.true_results = true_results
        self.predicted_results = predicted_results

    def youden_index(self, plot=1):
        """Calculates the youden_index for each threshold
        Takes no arguments except option to turn off plot
        Unless Specified, outputs a plot of Youden Index vs. thresholds
        """
        youden_index_values = []
        tpr = []
        fpt = []
        fpr, tpr, thresholds = set_threshold_roc_curve(self.true_results, self.predicted_results, pos_label=1, drop_intermediate=True)
        print thresholds
        for i in range(0, len(thresholds)):
            YI=((tpr[i]+(1-fpr[i])-1))/(math.sqrt(2))
            youden_index_values.append(YI)

        if(plot):
            plt.figure(figsize=(12, 9))
            ax = plt.subplot(111)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            plt.xticks(fontsize=19)
            plt.yticks(fontsize=19)
            plt.ylim([0.0, 0.5])
            plt.title('Optimal Accuracy Cutoff', fontsize=22)
            plt.xlabel('Classification Cutoff Threshold',fontsize=20)
            plt.ylabel('Youden Index',fontsize=20)
            plt.plot(thresholds, youden_index_values, color="#800000", linewidth=2)
            plt.show()

    def roc_curve(self, plot=1):
        """Plots the Reciever Operating Characteristic Curve
        Takes no arguments aside from option to turn off plot
        Unless specified, outputs a ROC curve
        """
        roc = roc_auc_score(self.true_results, self.predicted_results)
        fpr, tpr, thresholds = set_threshold_roc_curve(self.true_results, self.predicted_results, pos_label=1, drop_intermediate=True)

        if(plot):
            plt.figure(figsize=(12, 9))
            ax = plt.subplot(111)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            plt.xlim([0.0, 1.01])
            plt.ylim([0.0, 1.01])
            plt.xticks(fontsize=19)
            plt.yticks(fontsize=19)
            plt.plot(fpr, tpr, label='Area Under the Curve=%.2f' % roc, color="#800000", linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', label = 'Area Under the Random Guess Curve=0.5')
            plt.xlabel('1-specificity', fontsize=20)
            plt.ylabel('Sensitivity', fontsize=20)
            plt.title('Receiver Operating Characteristic Curve', fontsize=22)
            plt.legend(loc="lower right")
            plt.show()

    def well_rounded_validation(self):
        """Calculates a handful of important model validation metrics. I consider this a well rounded validation
        Takes no arguments
        Returns a Dict containing the AUROC, Recall, Precision, F1-score, Accuracy, and confusion matrix from the model
        """
        classified_predictions = classify(self.predicted_results, 0.5)
        conf_matrix = sklearn.metrics.confusion_matrix(self.true_results, classified_predictions, labels=None)

        return {
                "AUROC" : roc_auc_score(self.true_results, self.predicted_results),
                "Recall" : sklearn.metrics.recall_score(self.true_results, classified_predictions, labels=None, pos_label=1, average=None, sample_weight=None)[1],
                "Precision" : sklearn.metrics.precision_score(self.true_results, classified_predictions, labels=None, pos_label=1, average=None, sample_weight=None)[1],
                "F1 Score" : sklearn.metrics.f1_score(self.true_results, classified_predictions),
                "Accuracy" : sklearn.metrics.accuracy_score(self.true_results, classified_predictions),
                "Confusion Matrix" : [conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]]
                }

class visualize_data(object):
    """Offers an easy way to create beautiful histograms for the input data
    Takes a target value as constructor variable (enrichment values)
    """
    def __init__(self, enrichment):
        self.enrichment = enrichment
        self.target = classify(enrichment, 1.0)

    def continous_data_distribution(self, enrichment, particle):
        """This function creates a nice histogram of given data
        Takes a title and enrichment values as parameters
        outputs aesthetic graph
        """
        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.ylim([0.0, 250])
        plt.xlim([-3.0,3.0])
        plt.hist(np.log10(enrichment), bins=50, color = "#3F5D7D")
        plt.title('Histogram of '  + str(particle), y=1.08, fontsize=22)
        plt.ylabel('Frequency', fontsize=26)
        plt.xlabel('Logarithmic Enrichment Factor', fontsize=26)
        plt.tight_layout()
        plt.show()

    def visualize_by_particle(self):
        """Visualizes all the particle types in the dataset
        Takes no arguments
        Outputs 7 graphs, one for each reaction condition
        """
        self.continous_data_distribution(self.enrichment, 'Enrichment Factors on All Particles in The Database with 50 bins')
        self.continous_data_distribution(self.enrichment[0:356], 'Enrichment Factors on the Positive 10nm Silver Nanoparticle \n with no Solute')
        self.continous_data_distribution(self.enrichment[356:924], 'Enrichment Factors on the Negative 10nm Silver Nanoparticle \n with no Solute')
        self.continous_data_distribution(self.enrichment[924:1502], 'Enrichment Factors on the Negative 100nm Silver Nanoparticle \n with no Solute')
        self.continous_data_distribution(self.enrichment[1502:1989], 'Enrichment Factors on the Negative 10nm Silver Nanoparticle \n with 0.1mM Cysteine')
        self.continous_data_distribution(self.enrichment[1989:2499], 'Enrichment Factors on the Negative 10nm Silver Nanoparticle \n with 0.8 mM NaCl' )
        self.continous_data_distribution(self.enrichment[2499:3013], 'Enrichment Factors on the Negative 10nm Silver Nanoparticle \n with 3.0 mM NaCl')
        self.discrete_data_distribution()

    def scatterplot(self, data, x, y):
        """
        Takes in the dataframe and two columns of choice.
        Outputs a 2-d scatter plot of the data
        """
        bound_x = []
        bound_y = []
        unbound_x = []
        unbound_y = []
        for i, k in enumerate(self.target):
            if k == 0:
                bound_x.append(data[x][i])
                bound_y.append(data[y][i])
            else:
                unbound_x.append(data[x][i])
                unbound_y.append(data[y][i])

        line = plt.figure()

        plt.plot(unbound_y, unbound_x, "o", color='r', alpha=0.5)
        plt.plot(bound_y, bound_x, "o", color='g', alpha=0.5)
        plt.ylim([0, max(data[x])])
        plt.xlim([0, max(data[y])])
        plt.legend(('Bound', 'Unbound'), fontsize=18)
        plt.ylabel(str(x), fontsize = 26)
        plt.xlabel(str(y), fontsize=26)

def random_number():
    """This function imports the current time in nanoseconds to use as a pseudo-random number
    Takes no arguments
    Returns pseuo-random number
    """
    from datetime import datetime
    dt = datetime.now()
    rnum = dt.microsecond
    return rnum

def optimize(model, training_data, training_results):
    """This function optimizes the machine learning classifier, returning the parameters that give the best accuracy
    Takes the model, training data and training targets as arguments
    Outputs the best Parameters
    """
    #add whatever your heart desires to param grid, keep in mind its an incredibly inefficient algorithm
    param_grid = {
        'n_estimators': [1000],
        'max_features': ['auto'],
        'max_depth': [None],
        'min_samples_split': [5],
        'min_samples_leaf': [1],
        'n_jobs': [-1],
        'random_state' : [46, 0]
    }
    #5 fold validation
    CV_est = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5)
    CV_est.fit(training_data, training_results)
    print CV_est.best_params_

def recursive_feature_elimination(model, training_data, training_results):
    """Runs RFECV with 5 folds
    Takes model, training features, and targets as command line arguments
    Outputs optimum features
    """
    selector = RFECV(estimator=model, step=1, cv=5, scoring='roc_auc', verbose=1)
    selector = selector.fit(training_data, training_results)
    print selector.support_
    print "\n"
    print selector.ranking_
    print "\n"
    print "Optimal number of features: "
    print selector.n_features_
    print "\n"
    print selector.grid_scores_

def get_dummies(dataframe, category):
    """This function converts categorical variables into dummy variables
    Takes pandas dataframe and the catefory name as arguments
    Returns the dataframe with new dummy variables
    """
    dummy = pd.get_dummies(dataframe[category], prefix=category)
    dataframe = pd.concat([dataframe,dummy], axis = 1)
    dataframe.drop(category, axis=1, inplace=True)
    return dataframe

def classify(proba, cutoff):
    """This function classifies particles as bound or unbound
    Takes unclassified data the cutoff as arguments
    returns classified data in a list
    """
    predicted_results = []
    for i in proba:
        if i >= cutoff:
            temp = 1
        else:
            temp = 0
        predicted_results.append(temp)

    return predicted_results

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """
    Taken From sci-kit learn documentation to help set_threshold_roc_curve
    Altered to give a constant amount of thresholds for the roc curve
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def set_threshold_roc_curve(y_true, y_score, pos_label=None, sample_weight=None,
              drop_intermediate=True):
    """
    Taken from sci-kit learn documentation
    Altered to give a constant amount of thresholds for the roc curve
    """
    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    if tps.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]
    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]
    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]
    return fpr, tpr, thresholds

def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """
    Taken from sci-kit learn documentation to help set_threshold_roc_curve()
    Altered to return a constant amount of thrsholds for each roc curve
    Calculate true and false positives per binary classification threshold.
    """
    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)
    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (array_equal(classes, [0, 1]) or
             array_equal(classes, [-1, 1]) or
             array_equal(classes, [0]) or
             array_equal(classes, [-1]) or
             array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.
    # make y_true a boolean vector
    y_true = (y_true == pos_label)
    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.
    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        fps = stable_cumsum(weight)[threshold_idxs] - tps
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]

def clean_print(obj):
    """
    Prints the JSON in a clean format for all my
    Biochemistry friends
    """
    if type(obj) == dict:
        for key, val in obj.items():
            if hasattr(val, '__iter__'):
                print "\n" + key
                clean_print(val)
            else:
                print '%s : %s' % (key, val)
    elif type(obj) == list:
        for val in obj:
            if hasattr(val, '__iter__'):
                clean_print(val)
            else:
                print val
    else:
        print str(obj) + "\n"

def fetch_data():
    """
    Pulls the Data from CSV format. Returns 3012 measured protein-particle
    interactions represented as vectors
    """
    try:
        data = pd.read_csv("train.csv")
        target = pd.read_csv("class_result.csv")
        enrichment = pd.read_csv("result.csv")
    except:
        "Error Fetching CSV Data"
    #One hot encoding of categorical data
    data = get_dummies(data, 'size')
    data = get_dummies(data, 'charge')
    data = get_dummies(data, 'salt')
    data = get_dummies(data, 'cysteine')
    #Fill NaN's with average value in Abundance data
    count = 0
    total = 0
    for val in data['Abundance']:
        if not np.isnan(val):
            count+=1
            total+=val
    data = data.fillna(total/count)
    #Normalize the data
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    df_normalized = pd.DataFrame(np_scaled)
    #Classify enrichment data, using enrichment ratio of 1
    classed_enrich = []
    for i in enrichment.itertuples():
        if i[1] >= 1:
            temp = 1
        else:
            temp = 0
        classed_enrich.append(temp)
    #split data into training and testing set. Use testing set to validate model at the end
    training_data, test_data, training_results, test_results = sklearn.cross_validation.train_test_split(df_normalized, classed_enrich, test_size=0.1, random_state = random_number())
    #reshpae vectors
    training_results= np.ravel(training_results)
    test_results = np.ravel(test_results)
    enrichment = np.ravel(enrichment)
    target = np.ravel(target)

    return training_data, test_data, training_results, test_results, enrichment, target, data

def main():
    training_data, test_data, training_results, test_results, enrichment, target, data = fetch_data()
    #Visualize the data
    vis = visualize_data(enrichment)
    vis.visualize_by_particle()
    vis.scatterplot(data, 'Pi', 'Weight')
    #Print Relevant information
    print "Amount of Training data: " + str(len(training_data))
    print "Amount of Testing Data: " + str(len(test_data))

    est = RandomForestClassifierWithCoef(
                                 #criterion='mse',             #mean squared error criterion
                                 n_estimators=10000,             #number of trees used by the algorithm
                                 oob_score=True,               #Out of box score
                                 max_features='auto',          #features at each split (auto=all)
                                 max_depth=None,               #max tree depth
                                 min_samples_split=5,          #minimum amount of samples to split a node
                                 min_samples_leaf=1,           #minimum amount of samples a leaf can contain
                                 min_weight_fraction_leaf=0,   #minimum weight fraction of samples in a leaf
                                 max_leaf_nodes=None,          #maximum amount of leaf nodes
                                 n_jobs=-1,                    #CPU Cores used (-1 uses all)
                                 random_state=random_number()  #Initialize random seed generator
                                 )

    est.fit(training_data, training_results)                  #fit model to training data
    #Get prediction probabilities
    probability_prediction = est.predict_proba(test_data)[:,1]
    #Feature importance based on entropy calculations
    features = dict(zip(list(data), est.feature_importances_))
    #Run validation Metrics
    val = validation_metrics(test_results, probability_prediction)
    val.roc_curve()
    val.youden_index()
    return val.well_rounded_validation(), features

if __name__ == '__main__':
    results = {}
    for i in range(0, int(argv[1])):
        metrics = main()
        results["Run_" + str(i)] = metrics[0], metrics[1]
    print json.dumps(results)
