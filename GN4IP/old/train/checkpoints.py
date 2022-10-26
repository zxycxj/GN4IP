# classes for checkpoints

import time
import copy
import numpy as np
import torch
import scipy.io
from GN4IP.utils.save     import saveModel
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
    def evaluate(self, model, loader_va, loss_tr, loss_va, model_num=-1):
        
        # if at the right frequency
        if len(loss_tr) % self.frequency == 0:
            epoch = len(loss_tr)
            l_tr  = sum(loss_tr[-self.frequency: ]) / self.frequency * self.scale
            l_va  = sum(loss_va[-self.frequency: ]) / self.frequency * self.scale
            if self.log:
                l_tr = np.log(l_tr)
                l_va = np.log(l_va)
            message = "Epoch {:3n} | Training Loss {:.6f} | Validation Loss {:.6f}".format(epoch, l_tr, l_va)
            timeMessage(self.tstart, message)


# Create a subclass for stopping training early
class earlyStopping(checkpoint):
    
    # init
    def __init__(self, pat_reset=1, current_pat=1, min_loss=1e9, best_model=None, best_epoch=1, min_epochs=0, params2keep="best", other_name=""):
        super().__init__()
        self.name        = "earlyStopping"
        self.pat_reset   = pat_reset
        self.current_pat = pat_reset
        self.min_loss    = min_loss
        self.best_model  = best_model
        self.best_epoch  = best_epoch
        self.min_epochs  = min_epochs
        self.params2keep = params2keep
        self.other_name  = other_name
        
        # Create the reset dictionary
        self.create_reset_dict()
    
    # function for checking if training should stop
    def evaluate(self, model, loader_va, loss_tr, loss_va, model_num=-1):
        
        # If a new minimum loss value is achieved
        if loss_va[-1] <= self.min_loss:
            
            # Update the stopping information
            self.current_pat = self.pat_reset
            self.min_loss    = loss_va[-1]
            self.best_model  = copy.deepcopy(model.state_dict())
            self.best_epoch  = len(loss_va)
        
        # Else, decrease the patience
        else:
            self.current_pat -= 1


# Create a subclass for saving the model at checkpoints
class saveModelCheckpoint(checkpoint):
    
    # init
    def __init__(self, frequency=100, filename="no_name_model_e{}.pt"):
        super().__init__()
        self.name      = "saveModelCheckpoint"
        self.frequency = frequency
        self.filename  = filename
        
        # Create the reset dictionary
        self.create_reset_dict()
    
    # function for saving the model at a frequency
    def evaluate(self, model, loader_va, loss_tr, loss_va, model_num=-1):
        
        # if at the right frequency
        if len(loss_tr) % self.frequency == 0:
            
            # Create the filename based on model_num and epoch
            if model_num < 0:
                filename = self.filename.format(len(loss_tr))
            else:
                filename = self.filename.format(model_num, len(loss_tr))
            
            # save the model
            saveModel(model, filename)
            print("#  Saved a model checkpoint as "+filename)


# Create a subclass for saving predictions at checkpoints
class savePredictionCheckpoint(checkpoint):
    
    # init
    def __init__(self, frequency=1e6, sample_inds=[0], filename="no_name_model_e{}_pred.mat"):
        super().__init__()
        self.name        = "savePredictionCheckpoint"
        self.frequency   = frequency
        self.sample_inds = sample_inds
        self.filename    = filename
        
        # Create the reset dictionary
        self.create_reset_dict()
    
    # function for saving prediction at checkpoints
    def evaluate(self, model, loader_va, loss_tr, loss_va, model_num=-1):
        
        # If at the right frequency
        if len(loss_tr) % self.frequency == 0:
            
            # Put the model in eval mode
            model.eval()
            
            # Prepare storage lists
            x     = []
            dx    = []
            y     = []
            ypred = []
            
            # Loop through samples
            with torch.no_grad():
                for ind in self.sample_inds:
                    
                    # Use a gnn model on the sample
                    if model.type == "gnn":
                        x.append( loader_va.dataset[ind].x[:,0].squeeze().numpy())
                        dx.append(loader_va.dataset[ind].x[:,1].squeeze().numpy())
                        y.append( loader_va.dataset[ind].y[:,0].squeeze().numpy())
                        ypred_i = model(loader_va.dataset[ind].x.to(model.get_device()), loader_va.dataset[ind].edge_index.to(model.get_device()))
                    # use a cnn model on the sample
                    elif model.type == "cnn2d" or model.type == "cnn3d":
                        x.append( loader_va.dataset[ind][0][0,:].squeeze().numpy())
                        dx.append(loader_va.dataset[ind][0][1,:].squeeze().numpy())
                        y.append( loader_va.dataset[ind][1][0,:].squeeze().numpy())
                        ypred_i = model(loader_va.dataset[ind][0].unsqueeze(dim=0).to(model.get_device()))
                    
                    # bring sample back to cpu as numpy and append to list.
                    ypred.append(ypred_i.squeeze().cpu().detach().numpy())
            
            # Make into a dictionary to save
            save_dict = {
                "x"     : np.array( x    ),
                "dx"    : np.array(dx    ),
                "y"     : np.array( y    ),
                "ypred" : np.array( ypred)
            }
            
            # Create the filename based on model_num and epoch
            epoch = len(loss_tr) if self.frequency > 0 else self.frequency
            if model_num < 0:
                filename = self.filename.format(epoch)
            else:
                filename = self.filename.format(model_num, epoch)
            
            # Save the data
            scipy.io.savemat(filename, save_dict)
