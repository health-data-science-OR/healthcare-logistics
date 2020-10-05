# -*- coding: utf-8 -*-
"""
Provides simple functions for reading in TSP problems from known test problems.
Text file format from TSPLib: https://www.iwr.uni-heidelberg.de/groups/comopt/software/TSPLIB95/
Requires knowledge about the volumne of meta_data.

Slight issue - numpy.loadtxt does not like EOF. This is what is in the tsp files
my workaround is to delete EOF from the file!!
"""

import numpy as np

def read_coordinates(file_path, md_rows):
    """
    Return numpy array of city coordinates. 
    Excludes meta data (first 6 rows) and first column (city number)
    """
    return np.loadtxt(file_path, skiprows = md_rows, usecols = (1,2))

def read_meta_data(file_path, md_rows):
    """
    Returns meta dat/paraetmers a from specified tsp file
    """
    with open(file_path) as tspfile:
        return [next(tspfile).rstrip('\n') for x in range(md_rows - 1)]
        
        


    
    

