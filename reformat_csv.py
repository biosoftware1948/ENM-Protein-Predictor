"""Developed by: Joseph Pham 2020

This module is only used to reformat part of the CSV database file so that it's updated with new information
to use for further developing the random forest model

The output should be a new reformatted CSV file, so that if mistakes are made, the original is still there for use"""

import csv 
import os
import sys
import xlrd
import pandas as pd
import xlsxwriter

def updateHeaders(input_file, output_file, header_names):
    """ This function takes in an input file, output file, and a header_name array as arguments, 
    and rewrites a new file with updated headers so it's easier to manipulate and filter data afterwards"""
    with open(input_file, 'r', encoding='utf-8-sig') as fp:
        reader = csv.DictReader(fp, fieldnames=header_names)
        with open(output_file, 'w', newline='') as fh: 
            writer = csv.DictWriter(fh, fieldnames=reader.fieldnames)
            writer.writeheader()
            header_mapping = next(reader)
            writer.writerows(reader)
    return output_file

def updateColumnValues(comp1, comp2, output_file):
    """There are better ways to implement this overall, but this will have to suffice for now
    Params: 
    
    """
    print("nice")

def removeColumns(DataFrame, cols):
   #intended to remove empty/unnecessary columns if needed 
    print(cols)
    for i in range(38):
        if DataFrame.loc[cols + ": " + str(i)]:
            DataFrame = DataFrame.drop(columns=cols + i)
    return DataFrame

if __name__ == '__main__':
    """ 
    This module is intended to be used to replace headers and replace specific columns in a new duplicate file 
    INPUT: file to be manipulated and duplicated
    OUTPUT:output_file is the new CSV file that is returned 
    """
    assert len(sys.argv) == 2 
    #necessary files to manipulate
    output_file = 'Reformatted_Files/' + sys.argv[1]
    temp_file = 'Reformatted_Files/' + sys.argv[1]  #temporary file to manipulate before writing into the output file

    ##You can change which files whose information is needed, files will need to be cleanly structured (needs to be better defined)
    input_file = "Input_Files/database.csv" #Original file to copy contents from 
    bound_information = "Input_Files/information.csv" #Contains new information to include

    #declare new header row for new duplicate CSV file, can change or add new headers as you wish
    header_names = ['Accesion Number', 'pI', 'Protein Length', 'Protein Weight', 'Enzyme Commission Number',
        'Particle Size', 'Particle Charge', 'Solvent Cysteine Concentration', 'Solvent NaCl Concentration', 'Sequence', 
            '% Aromatic', '% Negative', '% Positive', '% Hydrophilic', '% Cysteine', 'Protein Abundance', 'Bound Fraction', 'Interprot']

    #update column names of temporary file that'll be used before writing to output file
    updateHeaders(input_file, temp_file, header_names)

    #we open them as dictionaries to filter by the necessary keys
    bound_information = list(csv.DictReader(open(bound_information, encoding='utf-8-sig')))
    temp_file = list(csv.DictReader(open(temp_file, encoding='utf-8-sig')))
    writer = csv.DictWriter(open(output_file, 'w'), fieldnames=header_names, extrasaction='ignore')
    writer.writeheader()

    #Filter by specific keys and insert new information in
    for row in bound_information:
        for r in temp_file:
            #could probably look up r's keys just by row's key, but for now this inefficient double for loop will suffice O(n*m)
            if row['Accession Number'] == r['Accesion Number']:
                for i in range(1,7):
                    ef = 'EF' + str(i)
                    df = 'DF' + str(i)
                    if row[ef] == r['Bound Fraction']:
                        r['Bound Fraction'] = row[df] ## actually just reformat and refilter by DF1 - DF6 and insert new values
                        writer.writerow(r)