# This script defines the trainingParameters class

import GN4IP

class trainingParameters(object):

    def __init__(self, scale=1, batch_size=1, learning_rate=0.001, device="cpu", max_epochs=1, checkpoints=[]):
        self.scale = scale
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.max_epochs = max_epochs
        self.checkpoints = checkpoints
        
    def printHeader(self):
        GN4IP.utils.printLine()
        print("#  Training a model")
        GN4IP.utils.printLine()
        print("#   Training On:", self.device)
        print("#    Batch Size:", self.batch_size)
        print("# Learning Rate:", self.learning_rate)
        print("#         Scale:", self.scale)
        print("#    Max Epochs:", self.max_epochs)
        
        