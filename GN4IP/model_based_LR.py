# Define a class used for model-based learned reconstruction

import torch
import GN4IP

class ModelBased(GN4IP.LearnedReconstruction):
    '''
    A ModelBased object is a learned reconstruction object with specific train
    and predict methods.
    '''
    
    def __init__(self, model_names, model=None):
        '''
        A PostProcess object is initialized the same as any 
        LearnedReconstruction object: with a list of model names.
        '''
        super().__init__(model_names, model)
    
    def train(self, data_tr, data_va, params_tr, Edges=None, Clusters=None, OVERWRITE=False):
        '''
        For each name in self.model_names, train self.model using the data
        and parameters provided. After training each model, save the parameters
        as self.model_names[i]. If the file in self.model_names[i] already 
        exists and OVERWRITE is False, don't retrain it, just load the 
        parameters into self.model and move on.
        '''
        
        # Add the params_tr, Edges, and Clusters to self
        self.params_tr = params_tr
        self.params_pr = params_tr # need params_pr.scale = params_tr.scale
        self.Edges = Edges
        self.Clusters = Clusters
        
        # Check if the model exists and/or should be trained
        TRAIN_MODELS = self.checkNames(OVERWRITE)
        
        # Initialize the training results object
        self.results_tr = GN4IP.train.TrainingResults()
        
        # For each name in model_names
        # for name in self.model_names:
        for i in range(len(self.model_names)):
            
            # Compute second input term using the update function
            if data_tr[1] is None:
                data_tr[1] = self.params_tr.updateFunction(data_tr[0], self.params_tr.fwd_data_file)
            if data_va[1] is None:
                data_va[1] = self.params_tr.updateFunction(data_va[0], self.params_tr.fwd_data_file)
            
            # If the model should be trained and saved
            if TRAIN_MODELS[i]:
                self.results_tr.appendResults(self.trainModel(data_tr, data_va, self.model_names[i]))
            
            # Or load the model
            else:
                self.model.load_state_dict(torch.load(self.model_names[i]))
                results_tr = GN4IP.train.trainingResults() 
                results_tr.loadedModel(self.model_names[0])
                self.results_tr.appendResults(results_tr)
            
            # Use the model to compute the next iterates
            data_tr[0] = self.predictModel(data_tr[0:-1]).predictions[0]
            data_va[0] = self.predictModel(data_va[0:-1]).predictions[0]
            data_tr[1] = None
            data_va[1] = None
    
    def predict(self, data_pr, params_pr, Edges=None, Clusters=None):
        '''
        For each name in model_names, the model is used to predict. The 
        function in self.params_pr.updateFunction are used to compute part of
        the input at each iteration. The model parameters are loaded from
        self.model_names at each iteration.
        '''
        
        # Add the params_tr, Edges, and Clusters to self
        self.params_pr = params_pr
        self.Edges = Edges
        self.Clusters = Clusters
        
        # If model is None, build it
        if self.model is None:
            self.model = GN4IP.models.build()
        
        # Check which models will be/can be loaded
        #self.checkNames(None)
        
        # Initialize the prediction results
        self.results_pr = GN4IP.predict.PredictionResults(data_pr[0], data_pr[1])
        
        # For each name in model_names
        for name in self.model_names:
            
            # Make sure the data is all there
            if data_pr[1] is None:
                data_pr[1] = self.params_pr.updateFunction(data_pr[0], self.params_pr.fwd_data_file)
            
            # Load the model parameters
            self.model.load_state_dict(torch.load(name))
            
            # Use the model to compute the next iterates
            self.results_pr.appendPredictions(self.predictModel(data_pr))
            
            # Prepare for the next iteration
            data_pr[0] = self.results_pr.predictions[-1]
            data_pr[1] = None
    