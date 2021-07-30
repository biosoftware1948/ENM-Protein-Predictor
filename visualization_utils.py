""" Developed by: Matthew Findlay 2017

This module contains all the visualization tools used to analyze our data.
"""
from __future__ import division
import data_utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


class visualize_data(object):
    """Offers an easy way to create visualizations for the input data
    Args:
        :param: data (pandas Dataframe): The X data from the database
        :param: bound_fraction (pandas Series): Target data from the X data
    Attributes:
        :data (pandas Dataframe): The X data
        :target (array of doubles): The data from the column 'Bound Fraction', from the X data
    """
    def __init__(self, data, bound_fraction):
        self.data = data
        self.target = np.array(bound_fraction)

    @staticmethod
    def describe_data():
        """Quickly visualize and describe the inputted dataset

        Args: None
        Returns:
            dataset_info.txt: a text file containing information about data types and structure of dataset
            dataset_numerical_attributes.csv: a csv file containing numerical attributes from dataset
        """
        # output information like dimensions and datatypes of the dataset
        f = open('Output_Files/data_information/dataset_info.txt', 'w+')
        v.data.info(buf=f)
        f.close()

        # output numerical attributes of dataset
        v.data.describe(exclude=[object]).to_csv("Output_Files/data_information/dataset_numerical_attributes.csv")

        # Creates a subplot grid, with "n" rows x 5 columns
        fig_hist, axs_hist = plt.subplots(figsize=(20, 15), nrows=math.ceil(len(v.data.columns)/5), ncols=5)
        a = axs_hist.ravel()

        # to prevent empty subplots, turn off all axs, and then turn them on when plotting
        for ax in a:
            ax.set_axis_off()

        # loop through each axes object and plot respective histogram on each one
        for i, ax in enumerate(a):
            # if there aren't enough histograms to fill the entire grid of subplots
            if i == len(v.data.columns):
                break
            ax.set_axis_on()
            ax.hist(v.data[v.data.columns[i]], bins=50)
            ax.set_xbound(lower=min(v.data[v.data.columns[i]]) - (.1 * v.data[v.data.columns[i]].mean()),
                          upper=max(v.data[v.data.columns[i]]) + (.1 * v.data[v.data.columns[i]].mean()))
            ax.set_xlabel(v.data.columns[i], fontsize=14)
            # allows for controlling font size on a set of y-axes
            if i % 5 == 0:
                ax.set_ylabel('Protein Count', fontsize=14)
            ax.grid(axis='both')

        # Extra formatting for dataset feature histograms
        plt.suptitle("Protein Count vs Dataset Features", fontsize=18)
        plt.tight_layout()
        fig_hist.savefig('Output_Files/data_information/histogram_dataset.png')

        # plot and save the target label (Bound Fraction)
        pd.DataFrame(v.target).hist(bins=50, figsize=(15, 15))
        plt.title('Protein Count v.s. Bound Fraction', fontsize=25)
        plt.ylabel('Protein Count', fontsize=25)
        plt.xlabel('Bound Fraction (Spectral Count)', fontsize=25)
        plt.savefig('Output_Files/data_information/bound_fraction.png')


def visualize_rfecv(grid_scores):
    """Plots the accuracy obtained with every number of features used from RFECV
    Args:
        :param: grid_scores (ndarray): contains cross-validation scores from RFECV
    Returns: a saved matplotlib graph of the rfecv_visualization
    """
    plt.figure(figsize=(16, 9))
    plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=25, fontweight='bold', pad=20)
    plt.xlabel('Number of features selected', fontsize=25, labelpad=20)
    plt.ylabel('Neg-MSE (Negative Mean Squared Error) Scoring', fontsize=25, labelpad=20)
    plt.plot(range(1, len(grid_scores) + 1), grid_scores, color='#303F9F', linewidth=3)
    plt.savefig('Output_Files/rfecv_visualization.png')
    plt.show()


def visualize_feature_importances(feature_importances):
    """ Plot the averaged feature importances from the most to least important as a bar graph
    Args:
        :param: feature_importances (dict): a dictionary containing the features and their corresponding averaged Gini
    importance value
    Returns: None
    """
    features = list(feature_importances.keys())
    values = list(feature_importances.values())

    plt.barh(features, values)
    plt.title('Averaged Gini Feature Importance Scores', fontsize=25)
    plt.ylabel('Features')
    plt.xlabel('Averaged Gini Feature Importance Score')

    plt.savefig('Output_Files/averaged_feature_importances.png')
    plt.show()


if __name__ == "__main__":
    db = data_utils.data_base()
    db.raw_data = "Input_Files/database.csv"
    bound_fraction_data = db.raw_data['Bound Fraction']
    db.clean_raw_data()

    # Call these functions to describe our dataset
    v = visualize_data(db.original, bound_fraction_data)
    v.describe_data()

