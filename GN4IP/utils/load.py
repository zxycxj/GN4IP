# Functions for loading data


import torch
import scipy.io
import numpy as np

# load a dataset from a .mat file
def loadDataset(filename, keys, verbose=0):

    # Use scipy.io to load the .mat file as a dictionary
    data = scipy.io.loadmat(filename)
    
    # Clean a few things up
    data["Edges"]    = data["Edges"][0]
    data["Centers"]  = data["Centers"][0]
    data["Clusters"] = data["Clusters"][0]
    
    # Subtract 1 from each of the edges and clusters
    # MATLAB starts at 1 while Python wants it to start at 0
    data["Edges"]    = [              edges-1  for edges    in data["Edges"]   ]
    data["Clusters"] = [np.squeeze(clusters-1) for clusters in data["Clusters"]]
    
    # Use the keys to select things to return
    output = []
    for key in keys:
        output.append(data[key])
    
    # return 
    return output
