""" Developed by: Matthew Findlay 2017

This module contains all the visualization tools used to analyze our data.
"""
from __future__ import division
import data_utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import math
import os
import sys

import estimator


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


def visualize_feature_importances(feature_importances, stddev):
    """ Plot the averaged feature importances from the most to least important as a bar graph
    Args:
        :param: feature_importances (dict): a dictionary containing the features and their corresponding averaged Gini
        importance value
        :param: stddev (dict): dictionary containing the features and corresponding standard deviations for calculating
        error bars
    Returns: None
    """
    # NOTE: Need to consolidate Solvent NaCl concentration, Solvent Cysteine, Nanomaterial Size, and Nanomaterial Charge
    # combine similar features to reduce redundancy
    nacl_solvent = ['Solvent NaCl Concentration_0.0', 'Solvent NaCl Concentration_0.8', 'Solvent NaCl Concentration_3.0']
    cys_solvent = ['Solvent Cysteine Concentration_0.0', 'Solvent Cysteine Concentration_0.1']
    enzyme_commission_number = ['Enzyme Commission Number_0', 'Enzyme Commission Number_1', 'Enzyme Commission Number_4'
                                , 'Enzyme Commission Number_5']
    nano_size = ['Particle Size_10', 'Particle Size_100']
    nano_charge = ['Particle Charge_0', 'Particle Charge_1']
    amino_acids = ['% Hydrophilic', '% Cysteine', '% Aromatic', '% Negative', '% Positive']

    combined_nacl = 0
    combined_nacl_stddev = 0
    for nacl_sol in nacl_solvent:
        combined_nacl += feature_importances.pop(nacl_sol)
        combined_nacl_stddev += stddev.pop(nacl_sol)
    feature_importances['Solvent NaCl Concentration'] = combined_nacl
    stddev['Solvent NaCl Concentration'] = combined_nacl_stddev

    combined_cys = 0
    combined_cys_stddev = 0
    for cys_sol in cys_solvent:
        combined_cys += feature_importances.pop(cys_sol)
        combined_cys_stddev += stddev.pop(cys_sol)
    feature_importances['Solvent Cysteine Concentration'] = combined_cys
    stddev['Solvent Cysteine Concentration'] = combined_cys

    combined_enzy = 0
    combined_enzy_stddev = 0
    for enzyme_num in enzyme_commission_number:
        combined_enzy += feature_importances.pop(enzyme_num)
        combined_enzy_stddev += stddev.pop(enzyme_num)
    feature_importances['Enzyme Commission Numbers'] = combined_enzy
    stddev['Enzyme Commission Numbers'] = combined_enzy_stddev

    combined_nano_size = 0
    combined_nano_size_stddev = 0
    for n_size in nano_size:
        combined_nano_size += feature_importances.pop(n_size)
        combined_nano_size_stddev += stddev.pop(n_size)
    feature_importances['Nanomaterial Size'] = combined_nano_size
    stddev['Nanomaterial Size'] = combined_nano_size_stddev

    combined_nano_charge = 0
    combined_nano_charge_stddev = 0
    for n_charge in nano_charge:
        combined_nano_charge += feature_importances.pop(n_charge)
        combined_nano_charge_stddev += stddev.pop(n_charge)
    feature_importances['Nanomaterial Charge'] = combined_nano_charge
    stddev['Nanomaterial Charge'] = combined_nano_charge_stddev

    for a_acid in amino_acids:
        key = a_acid + ' Amino Acids'
        feature_importances[key] = feature_importances.pop(a_acid)
        stddev[key] = stddev.pop(a_acid)

    feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1]))
    features = list(feature_importances.keys())
    values = list(feature_importances.values())

    # reorder standard deviation with respected to sorted feature importances + color code based on qualities
    protein = ['Protein Weight', 'Protein Abundance', 'pI', 'Enzyme Commission Numbers', '% H-bond/polar AA',
               '% Negative Amino Acids', 'GRAVY', '% Hydrophilic Amino Acids', '% Aromatic Amino Acids',
               '% Cysteine Amino Acids', '% Positive Amino Acids', '# of Metal Bonds']
    nanomaterial = ['Nanomaterial Charge', 'Nanomaterial Size']
    bar_colors = []

    for key in features:
        if key in protein:
            bar_colors.append('xkcd:teal')
        elif key in nanomaterial:
            bar_colors.append('xkcd:grey')
        else:
            bar_colors.append('xkcd:gold')
        stddev[key] = stddev.pop(key)

    f, ax = plt.subplots(figsize=(16,5))
    plt.barh(features, values, xerr=stddev.values(), color=bar_colors, ecolor='black', capsize=1.0)
    plt.title('Averaged Gini Feature Importance Scores', fontsize=20)
    plt.xlabel('Averaged Gini Feature Importance Score', fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()

    plt.savefig('Output_Files/averaged_feature_importances.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('script', help="script with which to run the model", type=str)
    parser.add_argument('-i', '--input', help="file path to your csv file to visualize distributions",
                        type=estimator.check_file_path, default="Input_Files/database.csv")
    args = vars(parser.parse_args())

    # initialize database and create useful distributions analyze
    db = data_utils.data_base()
    db.raw_data = args['input']
    bound_fraction_data = db.raw_data['Bound Fraction']
    db.clean_raw_data()

    # Call these functions to describe our dataset
    v = visualize_data(db.original, bound_fraction_data)
    v.describe_data()

