# classes for checkpoints

import time
import copy
import numpy as np
import torch
import scipy.io
from GN4IP.utils.message  import timeMessage


# Make a parent class for checkpoints
class checkpoint(object):

    # init
    def __init__(self):
        self.reset_dict = {}
        
    # Get attribute names to reset
    def get_var_names(self):
        var_names = list(vars(self).keys())
        var_names = var_names[1:]
        return var_names

    # Create the reset dictionary
    def create_reset_dict(self):
        var_names_to_reset = self.get_var_names()
        for name in var_names_to_reset:
            self.reset_dict[name] = getattr(self, name)
    
    # Set attribute values using the reset dictionary
    def reset(self):
        var_names_to_reset = self.get_var_names()
        for name in var_names_to_reset:
            setattr(self, name, self.reset_dict[name])
            
    # All subclasses should have a method evaluate() defined!!


# Create a subclass for printing loss values at checkpoints
class printLoss(checkpoint):
    
    # init
    def __init__(self, frequency=1, tstart=time.time(), scale=1, log=False):
        super().__init__()
        self.name      = "printLoss"
        self.frequency = frequency
        self.tstart    = tstart
        self.scale     = scale
        self.log       = log
        
        # Create the reset dictionary
        self.create_reset_dict()
    
    # function for printing the losses
    def evaluate(self, experiment):
    
        # if at the right frequency
        if len(experiment.results_tr.loss_tr) % self.frequency == 0:
            epoch = len(experiment.results_tr.loss_tr)
            l_tr  = sum(experiment.results_tr.loss_tr[-self.frequency: ]) / self.frequency * self.scale
            l_va  = sum(experiment.results_tr.loss_va[-self.frequency: ]) / self.frequency * self.scale
            if self.log:
                l_tr = np.log(l_tr)
                l_va = np.log(l_va)
            message = "Epoch {:3n} | Training Loss {:.6f} | Validation Loss {:.6f}".format(epoch, l_tr, l_va)
            timeMessage(self.tstart, message)
        
        return experiment

# Create a subclass to stop training early
class earlyStopping(checkpoint):
    
    # init
    def __init__(self, pat_reset=1, current_pat=1, min_loss_va=1e9, min_epochs=0, keep_min_loss_va_params=True):
        super().__init__()
        self.name = "earlyStopping"
        self.pat_reset   = pat_reset
        self.current_pat = pat_reset
        self.min_loss_va = min_loss_va
        self.min_epochs  = min_epochs
        self.keep_min_loss_va_params = keep_min_loss_va_params
        
        # Create the reset dictionary
        self.create_reset_dict()
        
    # Function for checking stopping criteria
    def evaluate(self, experiment):
        
        # If a new minimum loss value is achieved
        if experiment.results_tr.loss_va[-1] <= self.min_loss_va:
            
            # Update stopping information
            self.current_pat = self.pat_reset
            self.min_loss_va = experiment.results_tr.loss_va[-1]
            # Update training results
            experiment.results_tr.stop_model = copy.deepcopy(experiment.model.state_dict())
            experiment.results_tr.stop_epoch = len(experiment.results_tr.loss_va)
        
        # Else, decrease the patience
        else:
            self.current_pat -= 1
            
            # If patience has run out (=0), send a stop indicator
            if self.current_pat <= 0: experiment.stop = True
            
            # if not keeping the parameters at the minimum validation loss, update training results
            if not self.keep_min_loss_va_params:
                experiment.results_tr.stop_model = copy.deepcopy(experiment.model.state_dict())
                experiment.results_tr.stop_epoch = len(experiment.results_tr.loss_va)
                
        return experiment



