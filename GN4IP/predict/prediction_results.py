# Create a class for prediction results

class PredictionResults(object):
    '''
    This object makes storing and accessing predictions from a model easier.
    '''
    
    def __init__(self, input, input_cnn=None):
        '''
        GNN results should be initialized with predictions (and updates if it
        is a model-based network). CNN results should be initialized with
        predictions and predictions_cnn (and the updates if it is a model-
        based network). Whatever isn't provided at initialization can be fixed
        later too.
        '''
        self.input = input
        self.input_cnn = input_cnn
        self.predictions = []
        self.updates = []
        self.predictions_cnn = []
        self.updates_cnn = []
        self.time_pr = []
    
    def appendPredictions(self, results_pr):
        '''
        Append a separate PredictionResults object to this one.
        '''
        self.predictions.extend(results_pr.predictions)
        self.updates.extend(results_pr.updates)
        self.predictions_cnn.extend(results_pr.predictions_cnn)
        self.updates_cnn.extend(results_pr.updates_cnn)
        self.time_pr.extend(results_pr.time_pr)