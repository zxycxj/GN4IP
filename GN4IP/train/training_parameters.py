# Create a class for training parameters

import GN4IP

class TrainingParameters(object):
    '''
    This object makes storing and accessing training parameters easier.
    '''
    
    def __init__(self, device="cpu", scale=1, batch_size=2, learning_rate=0.001, max_epochs=10, checkpoints=[], updateFunction=None, fwd_data_file=""):
        '''
        Initialize the parameters object
        '''
        self.device = device
        self.scale = scale
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.checkpoints = checkpoints
        self.updateFunction = updateFunction
        self.fwd_data_file = fwd_data_file
    
    def printHeader(self):        
        '''
        Print some information about the training parameters.
        '''
        print("#  Training a model")
        GN4IP.utils.printLine()
        print("#   Training On:", self.device)
        print("#    Batch Size:", self.batch_size)
        print("# Learning Rate:", self.learning_rate)
        print("#         Scale:", self.scale)
        print("#    Max Epochs:", self.max_epochs)
