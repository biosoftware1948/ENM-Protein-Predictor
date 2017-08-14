"""Developed by: Matthew Findlay 2017

This module parses the JSON output from the estimator script and prints it
in readable format to the command line.

This can be run by passing a json file as a command line argument.
"""
import data_utils
import numpy as np
import json
import operator
import sys

#Open json file
with open(sys.argv[1], 'r') as json_data:
    data = json.load(json_data)
#Dictionaries to hold information
average_metrics = {}
average_importances = {}
std_metrics = {}
std_importances = {}
confusion_matrix = np.zeros(shape=(4,))
runs = 0
#pull information
for run in data:
    runs+=1
    #Grab val metrics
    for key, value in data[run]['scores'].items():
        if key == 'Confusion Matrix':
            confusion_matrix = np.add(confusion_matrix, value)
        elif key in average_metrics:
            average_metrics[key] = average_metrics[key] + value
            std_metrics[key].append(value)
        else:
            average_metrics[key] = value
            std_metrics[key] = [value]
    #Grab feature importances
    for key, value in data[run]['importances'].items():
        if key in average_importances:
            average_importances[key] = average_importances[key] + value
            std_importances[key].append(value)
        else:
            average_importances[key] = value
            std_importances[key] = [value]

#Divide average values to get mean
for key, value in average_metrics.items():
    average_metrics[key] = value/runs
for key, value in average_importances.items():
    average_importances[key] = value/runs
#Print
print "#########Confusion Matrix over all 50 runs#########"
print "unbound: {}, {}\nbound: {}, {}\n".format(confusion_matrix[0],confusion_matrix[1],confusion_matrix[2],confusion_matrix[3])

print "#########VALUES########"
print "\n~Performance Metrics~"
data_utils.clean_print(average_metrics)
print "\n~Feature Importances~"
sorted_imports = sorted(average_importances.items(), key=operator.itemgetter(1))
for tup in reversed(sorted_imports):
    print "{}: {}".format(tup[0], tup[1])

#get std results for metrics
print "\n########STD VALUES#######"
print "\n~Performance Metrics~"
for key, arr in std_metrics.items():
    arr = np.std(np.array(arr))
    std_metrics[key] = arr
for key, arr in std_metrics.items():
    print "{}: {}".format(key, arr)
#GEt std results for importances
print "\n~Feature Importances~"
for key, arr in std_importances.items():
    arr = np.std(np.array(arr))
    std_importances[key] = arr
for key, arr in std_importances.items():
    print "{}: {}".format(key, arr)
