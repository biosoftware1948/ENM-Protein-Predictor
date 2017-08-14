""" Developed by: Matthew Findlay 2017

This module contains all the visualization tools used to analyze our data.
"""
from __future__ import division
import data_utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from mpl_toolkits.mplot3d import Axes3D
from string import ascii_letters

class visualize_data(object):
    """Offers an easy way to create visualizations for the input data

    Args:
        :param data (pandas Dataframe): The X data from the database
        :param enrichment (array of floats): The continous Y data from the
        database
    Attributes:
        :data (pandas Dataframe): The X data
        :enrichemnt (array of floats): The continous Y data
        :target (array of ints): The classified Y data
    """
    def __init__(self, data, enrichment):
        self.data = data
        self.enrichment = enrichment
        self.target = data_utils.classify(enrichment, 0.5)

    def interactive_3d_plot(self, x_label, y_label, z_label):
        """Creates a cool interactive 3d plot the user can spin and to
        visualize data in 3d.

        Args:
            :param x_label (string): The data column for the x-axis
            :param y_label (string): The data column for the y-axis
            :param z_label (string): The data column for the z-axis
        Returns:
            None
        """
        assert isinstance(self.data, pd.DataFrame), "Please pass first argument as pandas dataframe"
        assert all(isinstance(i, str) for i in [x_label, y_label, z_label]), "labels must be strings"
        bound = 1
        unbound = 0
        x = np.array(self.data[x_label])
        y = np.array(self.data[y_label])
        z = np.array(self.data[z_label])

        plot_data = {'x': {'bound' : [], 'unbound': []},
                     'y': {'bound' : [], 'unbound': []},
                     'z': {'bound' : [], 'unbound': []}}

        for i, val in enumerate(self.target):
            if val==bound:
                plot_data['x']['bound'].append(x[i])
                plot_data['y']['bound'].append(y[i])
                plot_data['z']['bound'].append(z[i])
            else:
                plot_data['x']['unbound'].append(x[i])
                plot_data['y']['unbound'].append(y[i])
                plot_data['z']['unbound'].append(z[i])

        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')

        z_offset = 0
        ax.plot(plot_data['x']['bound'], plot_data['y']['bound'], plot_data['z']['bound'],'ko', alpha=0.4, label='Bound')
        ax.plot(plot_data['x']['unbound'], plot_data['y']['unbound'], plot_data['z']['unbound'],'ro', alpha=0.4, label='Unbound')

        ax.set_xlim3d(np.amin(x), np.amax(x))
        ax.set_ylim3d(np.amin(y), np.amax(y))
        ax.set_zlim3d(np.amin(z), np.amax(z))

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)

        ax.legend()
        plt.show()

    def continous_data_distribution(self, particle_name, enrichment=None):
        """This function creates a histogram of given data

        Args:
            :param particle_name (string): The name of the particle type
            :param enrichment (np.array of floats): The X data to be binned
        Returns:
            None
        """
        if enrichment is None:
            enrichment = self.enrichment

        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.ylim([0.0, 100])
        plt.xlim([-3.0,3.0])
        plt.hist(np.log10(enrichment), bins=25, color = "#3F5D7D")
        plt.title('Histogram of '  + str(particle_name), y=1.08, fontsize=22)
        plt.ylabel('Frequency', fontsize=26)
        plt.xlabel('Logarithmic Enrichment Factor', fontsize=26)
        plt.tight_layout()
        plt.show()

    def continous_distribution_by_particle(self):
        """Visualizes all the particle types in the dataset,
        Outputs 7 graphs, one for each reaction condition.

        Args:
            None
        Returns:
            None
        """
        self.continous_data_distribution('Enrichment Factors on All Particles in The Database with 50 bins')
        self.continous_data_distribution('Enrichment Factors on the Positive 10nm Silver Nanoparticle \n with no Solute', self.enrichment[0:356])
        self.continous_data_distribution('Enrichment Factors on the Negative 10nm Silver Nanoparticle \n with no Solute', self.enrichment[356:924])
        self.continous_data_distribution('Enrichment Factors on the Negative 100nm Silver Nanoparticle \n with no Solute', self.enrichment[924:1501])
        self.continous_data_distribution('Enrichment Factors on the Negative 10nm Silver Nanoparticle \n with 0.1mM Cysteine', self.enrichment[1501:1988])
        self.continous_data_distribution('Enrichment Factors on the Negative 10nm Silver Nanoparticle \n with 0.8 mM NaCl', self.enrichment[1988:2498] )
        self.continous_data_distribution('Enrichment Factors on the Negative 10nm Silver Nanoparticle \n with 3.0 mM NaCl', self.enrichment[2499:3011])
        self.discrete_data_distribution()

    def scatterplot(self, x, y):
        """
        Outputs a 2-d scatter plot of the data

        Args:
            :param x (string): The name of the column desired for x-axis
            :param y (string): The name of the column desired for the y-axis
        Returns:
            None
        """
        plot_data = {'x': {'bound' : [], 'unbound': []},
                     'y': {'bound' : [], 'unbound': []}}

        for i, k in enumerate(self.target):
            if k == 1:
                plot_data['x']['bound'].append(self.data[x][i])
                plot_data['y']['bound'].append(self.data[y][i])
            else:
                plot_data['x']['unbound'].append(self.data[x][i])
                plot_data['x']['unbound'].append(self.data[y][i])

        line = plt.figure()
        plt.plot(plot_data['x']['unbound'], plot_data['x']['unbound'], "o", color='r', alpha=0.5)
        plt.plot(plot_data['y']['bound'].append(data[y][i]), plot_data['x']['bound'], "o", color='g', alpha=0.5)
        plt.ylim([0, max(self.data[x])])
        plt.xlim([0, max(self.data[y])])
        plt.legend(('Bound', 'Unbound'), fontsize=18)
        plt.ylabel(str(x), fontsize = 26)
        plt.xlabel(str(y), fontsize=26)


    def discrete_data_distribution(self):
        """This function gives a visualization of class balance in the data

        Args:
            None
        Returns:
            None
        """
        bound = 0
        ubound = 0
        iterations = 0
        for i in self.target:
            if i == 1:
                bound = bound + 1
            else:
                ubound = ubound + 1

        #Plot
        x = [0,1]
        y = [ubound, bound]
        plt.bar(x, y, width=0.1, color='blue')
        plt.title('Class Counts')
        plt.ylabel('Frequency')
        plt.xlabel('Class')
        plt.show()

    def autolabel(self, rects, ax):
        """
        Attach a text label above each bar displaying its height.

        Args:
            :param rects (list): rectangles to label
            :param ax (obj): The axis of the pyplot graph
        Returns:
            None
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom', fontsize=20)

    @staticmethod
    def roc_plot(roc, fpr, tpr, thresholds):
        """Generates a roc_plot (true positive rate vs false positive rate)
        and shows the area under the curve in the legend

        Args:
            :param roc (float): The area under the roc curve
            :param fpr (np array of floats): The false positive rate for each
            threshold
            :param tpr (np array of floats): The true positive rate for each
            threshold
            :param thresholds (np array of floats): The thresholds that lead
            to the tpr and fpr
        Returns:
            None
        """
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

    @staticmethod
    def youden_index_plot(thresholds, youden_index_values):
        """Generates a plot of youden_index vs thresholds

        Args:
            :param thresholds (np array of floats): The thresholds used
            :param youden_index_values (np array of floats): The values of
            the youden index at each threshold
        Returns:
            None
        """
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

    def correlation_plot(self):
        """Plots a correlation plot of all the continous variables in the
        database. This plot helped us remove protein length which was highly
        correlated with protein weight.

        Args, Returns:  None
        """
        corr = self.data.iloc[:,:8].corr()
        f, ax = plt.subplots(figsize=(11,9))
        ax.grid(False)
        for (i, j), z in np.ndenumerate(corr):
            ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

        plt.imshow(corr, cmap='RdYlGn', interpolation='none', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(ticks = (-1.0, -0.5, 0, 0.5, 1))
        plt.xticks(range(len(corr)), corr.columns, rotation='vertical', fontsize=20)
        plt.yticks(range(len(corr)), corr.columns, fontsize=20);
        #plt.suptitle('Correlation plot of protein features', fontsize=22, fontweight='bold')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    db = data_utils.data_base()
    db.raw_data = "Input_Files/database.csv"
    db.clean_raw_data()
    v = visualize_data(db.clean_X_data, db.Y_enrichment)
    v.correlation_plot()
    v.continous_distribution_by_particle()
