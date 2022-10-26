# Loss functions for training models

import numpy as np
import torch



class LossesBilly(object):
    
    def __init__(self, weights=None):
        self.weights = weights
    
    def tileWeights(self, input, n_samples):
        if n_samples > 1:
            dims = [1]*len(input.size())
            dims[0] = n_samples
            weights = torch.tile(self.weights, dims)
        else:
            weights = self.weights
        return weights


class myMSELoss(LossesBilly):
    
    def __init__(self, weights=None):
        super().__init__(weights)
    
    def evaluate(self, input, target, n_samples=1):
        
        # Equal weights
        if self.weights is None:
            loss = ((input - target) ** 2).mean()
        
        # Custom weights
        else:
            weights = self.tileWeights(input, n_samples)
            loss = (weights * (input - target) ** 2).mean()
        
        return loss


class myL1Loss(LossesBilly):
    
    def __init__(self, weights=None):
        super().__init__(weights)
    
    def evaluate(self, input, target, n_samples=1):
        
        # Equal weights
        if self.weights is None:
            loss = abs(input - target).mean()
        
        # Custom weights
        else:
            weights = self.tileWeights(input, n_samples)
            loss = (weights * abs(input - target)).mean()
        
        return loss


class mySmoothL1Loss(LossesBilly):
    
    def __init__(self, weights=None, beta=1):
        super().__init__(weights)
        self.beta = beta
    
    def evaluate(self, input, target, n_samples=1):
        
        # Equal weights
        if self.weights is None:
            if abs(input-target).mean() < self.beta:
                loss = (0.5 * (input - target) ** 2 / self.beta).mean()
            else:
                loss = (abs(input - target) - 0.5 * self.beta).mean()
        
        # Custom weights
        else:
            weights = self.tileWeights(input, n_samples)
            if (weights * abs(input-target)).mean() < self.beta:
                loss = (0.5 * (weights * abs(input-target)) ** 2 / self.beta).mean()
            else:
                loss = ((weights * abs(input-target)) - 0.5 * self.beta).mean()
        
        return loss