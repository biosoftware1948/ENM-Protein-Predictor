from __future__ import division

import data_utils
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import math


class visualize_data(object):
    """Offers an easy way to create beautiful histograms for the input data
    Takes a target value as constructor variable (enrichment values)
    """
    def __init__(self, enrichment):
        self.data = data
        self.enrichment = enrichment
        self.target = data_utils.classify(enrichment, 0.5)

    def interactive_3d_plot(self, data, x_label, y_label, z_label):
        """Creates a cool interactive 3d plot the user can spin and to
        visualize data in 3d.
        parameters : pandas dataframe containing data and three labels
        to visualize
        returns nothing
        """
        assert isinstance(data, pd.DataFrame), "Please pass first argument as pandas dataframe"
        assert all(isinstance(i, str) for i in [x_label, y_label, z_label]), "labels must be strings"
        bound = 1
        unbound = 0
        x = np.array(data[x_label])
        y = np.array(data[y_label])
        z = np.array(data[z_label])

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

    def continous_data_distribution(self, particle, enrichment=None):
        """This function creates a dank histogram of given data
        Takes a title and enrichment values as parameters
        outputs aesthetic graph
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
        self.continous_data_distribution('Enrichment Factors on All Particles in The Database with 50 bins')
        self.continous_data_distribution('Enrichment Factors on the Positive 10nm Silver Nanoparticle \n with no Solute', self.enrichment[0:356])
        self.continous_data_distribution('Enrichment Factors on the Negative 10nm Silver Nanoparticle \n with no Solute', self.enrichment[356:924])
        self.continous_data_distribution('Enrichment Factors on the Negative 100nm Silver Nanoparticle \n with no Solute', self.enrichment[924:1502])
        self.continous_data_distribution('Enrichment Factors on the Negative 10nm Silver Nanoparticle \n with 0.1mM Cysteine', self.enrichment[1502:1989])
        self.continous_data_distribution('Enrichment Factors on the Negative 10nm Silver Nanoparticle \n with 0.8 mM NaCl', self.enrichment[1989:2499] )
        self.continous_data_distribution('Enrichment Factors on the Negative 10nm Silver Nanoparticle \n with 3.0 mM NaCl', self.enrichment[2499:3013])
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


    def discrete_data_distribution(self):
        """This function gives a visualization of class balance in the data
        No input
        Output graph isnt as dank as the histogram but thats ok
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
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom', fontsize=20)
