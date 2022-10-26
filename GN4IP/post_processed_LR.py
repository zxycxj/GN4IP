# Define a class used for post-processed learned reconstruction

import torch
import GN4IP

class PostProcess(GN4IP.LearnedReconstruction):
    '''
    A PostProcess object is a learned reconstruction object with specific train
    and predict methods
    '''
    
    def __init__(self, model_names, model=None):
        '''
        A PostProcess object is initialized the same as any 
        LearnedReconstruction object
        '''
        super().__init__(model_names, model)

    
    def train(self, data_tr, data_va, params_tr, Edges=None, Clusters=None, OVERWRITE=False):
        '''
        Train self.model using the data and parameters provided. After 
        training, save the parameters as self.model_names[0]. If the file in
        self.model_names[0] already exists and OVERWRITE is False, don't 
        retrain.
        '''
        
        # Add things to self
        self.params_tr = params_tr
        self.Edges = Edges
        self.Clusters = Clusters
        
        # Check if the model exists and/or should be trained
        TRAIN_MODELS = self.checkNames(OVERWRITE)
        
        # Train the model
        if TRAIN_MODELS[0]:
            self.results_tr = self.trainModel(data_tr, data_va, self.model_names[0])
        
        # Or load the model
        else:
            self.model.load_state_dict(torch.load(self.model_names[0]))
            self.results_tr = GN4IP.train.trainingResults()
            self.results_tr.loadedModel(self.model_names[0])
        
        # Don't need to return anything
    
    def predict(self, data_pr, params_pr, Edges=None, Clusters=None):
        '''
        Use self.model to predict on data_pr using the parameters provided.
        The prediction results are stored in self.results_pr.
        '''
        
        # Add things to self
        self.params_pr = params_pr
        self.Edges = Edges
        self.Clusters = Clusters
        
        # Check if the model exists
        PREDICT_MODELS = self.checkNames(OVERWRITE)
        
        # If the models exist,
        if PREDICT_MODELS[0]:
            
            # Load the model parameters
            self.model.load_state_dict(torch.load(self.model_names[0]))
            
            # Predict using the model
            self.results_pr = self.predictModel(data_pr)
        
        # Don't need to return anything

