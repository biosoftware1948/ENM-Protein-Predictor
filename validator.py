import data_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import visualization_tools
import sklearn

class validation_metrics(object):
    """Several statistical tests used to validate the models predictive power
    Takes True results of the test data and the models results as parameters for constructor
    """
    def __init__(self, true_results, predicted_results):
        self.true_results = true_results
        self.predicted_results = predicted_results

    def youden_index(self):
        """Calculates the youden_index for each threshold
        Takes no arguments except option to turn off plot
        Unless Specified, outputs a plot of Youden Index vs. thresholds
        """
        tpr = []
        fpt = []
        fpr, tpr, thresholds = set_threshold_roc_curve(self.true_results, self.predicted_results, pos_label=1, drop_intermediate=True)

        youden_index_values = np.zeros([len(thresholds)])
        for i in range(0, len(thresholds)):
            youden_index_values[i] = ((tpr[i]+(1-fpr[i])-1))/(math.sqrt(2))

        visualization_tools.visualize_data.youden_index_plot(thresholds, youden_index_values)

    def roc_curve(self):
        """Plots the Reciever Operating Characteristic Curve
        Takes no arguments aside from option to turn off plot
        Unless specified, outputs a ROC curve
        """
        roc = sklearn.metrics.roc_auc_score(self.true_results, self.predicted_results)
        fpr, tpr, thresholds = set_threshold_roc_curve(self.true_results, self.predicted_results, pos_label=1, drop_intermediate=True)

        visualization_tools.visualize_data.roc_plot(roc, fpr, tpr, thresholds)

    def well_rounded_validation(self):
        """Calculates a handful of important model validation metrics. I consider this a well rounded validation
        Takes no arguments
        Returns a Dict containing the AUROC, Recall, Precision, F1-score, Accuracy, and confusion matrix from the model
        """
        classified_predictions = data_utils.classify(self.predicted_results, 0.5)
        conf_matrix = sklearn.metrics.confusion_matrix(self.true_results, classified_predictions, labels=None)

        return {
                "AUROC" : sklearn.metrics.roc_auc_score(self.true_results, self.predicted_results),
                "Recall" : sklearn.metrics.recall_score(self.true_results, classified_predictions, labels=None, pos_label=1, average=None, sample_weight=None)[1],
                "Precision" : sklearn.metrics.precision_score(self.true_results, classified_predictions, labels=None, pos_label=1, average=None, sample_weight=None)[1],
                "F1 Score" : sklearn.metrics.f1_score(self.true_results, classified_predictions),
                "Accuracy" : sklearn.metrics.accuracy_score(self.true_results, classified_predictions),
                "Confusion Matrix" : [conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]]
                }

#***HELPER FUNCTIONS***#
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
