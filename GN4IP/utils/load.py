# Functions for loading data


import torch
import scipy.io


# load a dataset from a .mat file
def loadDataset(filename, keys, verbose=0):

    # Use scipy.io to load the .mat file as a dictionary
    data = scipy.io.loadmat(filename)
    
    # Clean a few things up
    data["Edges"]    = data["Edges"][0]
    data["Centers"]  = data["Centers"][0]
    data["Clusters"] = data["Clusters"][0]
    
    # Use the keys to select things to return
    output = []
    for key in keys:
        output.append(data[key])
    
    # return 
    return output
