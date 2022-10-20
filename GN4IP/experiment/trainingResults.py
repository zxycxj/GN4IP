# This script defines the trainingResults class

import GN4IP

class trainingResults(object):
    
    def __init__(self):
        self.loss_tr = []
        self.loss_va = []
        self.max_epochs = -1
        self.stop_epoch = -1
        self.stop_model = None
        self.time_tr = 0
        
    def reset(self):
        self.loss_tr = []
        self.loss_va = []
        self.max_epochs = -1
        self.stop_epoch = -1
        self.stop_model = None
        self.time_tr = 0
    
    def printSummary(self):
        
        # Get the time in hrs:mins:secs
        if self.time_tr == 0:
            hrs  = 0
            mins = 0
            sec  = 0
        else:
            hrs  = int( self.time_tr//3600)
            mins = int((self.time_tr -3600*hrs)//60)
            sec  = int((self.time_tr -3600*hrs - 60*mins) // 1)
        
        GN4IP.utils.printLine()
        print("#  Training Completed in ({:02d}:{:02d}:{:02d})".format(hrs, mins, sec))
        GN4IP.utils.printLine()
        print("#            Total Epochs:", len(self.loss_tr))
        print("#   Model Stored at Epoch:", self.stop_epoch)
        print("#       Min Tr Loss Epoch:", self.loss_tr.index(min(self.loss_tr))+1)
        print("#       Min Va Loss Epoch:", self.loss_va.index(min(self.loss_va))+1)
        print("#   Minimum Training Loss:", min(self.loss_tr))
        print("# Minimum Validation Loss:", min(self.loss_va))