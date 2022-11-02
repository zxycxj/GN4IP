# Create a class for training results

import numpy as np
import scipy.io

class TrainingResults(object):
    '''
    This object makes storing and accessing training results easier.
    '''
    
    def __init__(self):
        '''
        Results are initialized as empty lists.
        '''
        self.names   = []
        self.loss_tr = []
        self.loss_va = []
        self.time_tr = []
        self.stop_model = None
        self.stop_epoch = 0
        self.stop = False
    
    def loadedModel(self, name):
        '''
        A model was loaded so just append the name and placeholders.
        '''
        self.names.append(name)
        self.loss_tr.append([])
        self.loss_va.append([])
        self.time_tr.append(0)
    
    def appendResults(self, results_tr):
        '''
        Append a separate TrainingResults object to this one. This is done when
        storing training results from multiple models in one object. Add the
        lists from the new results to the end of this object's results lists
        '''
        self.names.extend(results_tr.names)
        self.loss_tr.extend(results_tr.loss_tr)
        self.loss_va.extend(results_tr.loss_va)
        self.time_tr.extend(results_tr.time_tr)
    
    def save(self, name):
        '''
        Save the results using the name provided. Put things into a dictionary
        so they can be saved as a .mat file.
        '''
        save_dict = {
            "training_losses"   : self.makeArrayFromRaggedLists(self.loss_tr),
            "validation_losses" : self.makeArrayFromRaggedLists(self.loss_va),
            "training_times"    : np.array(self.time_tr),
            "model_names"       : self.names,
        }
        scipy.io.savemat(name, save_dict)
    
    def makeArrayFromRaggedLists(self, list_of_ragged_lists, filler=0):
        '''
        Return a np.array from the list_of_ragged_lists. The list of ragged
        lists could be something like [[1,2],[1,2,3]] which cannot be simply
        turned into an np.array by saying np.array(list_of_ragged_lists).
        '''
        # Determine the max length of list
        max_len = 0
        for sublist in list_of_ragged_lists:
            max_len = max(max_len, len(sublist))
        
        # Fill the shorter lists with the filler on the end
        for i in range(len(list_of_ragged_lists)):
            list_of_ragged_lists[i] = list_of_ragged_lists[i] + [filler]*(max_len-len(list_of_ragged_lists[i]))
        
        # Convert the list of now non-ragged lists to an np.array
        return np.array(list_of_ragged_lists)
    