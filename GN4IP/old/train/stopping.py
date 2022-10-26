# Functions for stopping training

import copy
import torch

# Stop early if validation loss fails to decrease
def earlyStopping(stopping_info, model, l_va, epoch):
    
    # stopping_info has the following
    # [0] current patience
    # [1] patience reset value
    # [2] minimum validation loss value
    # [3] state_dict() of best model
    # [4] best epoch
    
    # If a new minimum validation loss is reached
    if l_va <= stopping_info[2]:
        stopping_info[0] = stopping_info[1]
        stopping_info[2] = l_va
        stopping_info[3] = copy.deepcopy(model.state_dict()) # Need a deep copy here
        stopping_info[4] = epoch
        #print("Best")
        #print(stopping_info[3][next(iter(stopping_info[3]))])
    
    # If a new minimum validation loss is not reached
    else:
        stopping_info[0] -= 1
        
    return stopping_info
