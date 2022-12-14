# Make a parent class for the ModelBased and PostProcess classes

import time
import numpy as np
import torch
import GN4IP
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class LearnedReconstruction(object):
    '''
    This is a parent class that will help build the ModelBased and PostProcess
    classes. Both of those classes share these same methods but have their own 
    train and predict methods.
    '''
    
    def __init__(self, model_names, model=None):
        '''
        Initialize the LearnedReconstruction object with a list of names to 
        save the model(s) as and a model architecture to use. The object will
        also be initialized with training and prediction parameter and result
        attributes.
        '''
        self.model_names = model_names
        self.params_tr = None
        self.results_tr = None
        self.params_pr = None
        self.results_pr = None
        
        # If a model isn't provided, use the default one
        if model is None:
            self.model = GN4IP.models.buildModel()
        else:
            self.model = model
    
    def train(self, data_tr, data_va, params_tr, Edges=None, Clusters=None, OVERWRITE=False):
        ''''
        The specific reconstruction methods will need to define their own
        training routines.
        '''
        pass
    
    def predict(self, data_pr, params_pr, Edges=None, Clusters=None):
        '''
        The specific reconstruction methods will need to define their own 
        prediction routines.
        '''
        pass
    
    def trainModel(self, data_tr, data_va, name):
        '''
        Train self.model with the training and validation data provided and
        according to the self.params_tr. The data is passed as graph data where
        data[0:-1] is a list of [N by M] np.arrays describing the input 
        features and data[-1] is an [N by M] np.array containing the ground 
        truth. Return a TrainingResults object that includes the loss values 
        and training time.
        '''
        
        # Assign self.Edges and self.Clusters to self.model as tensors if not None
        if self.Edges is not None:
            self.model.Edges    = [torch.as_tensor(e.astype(int)).squeeze().to(self.params_tr.device) for e in self.Edges[1:]]
        if self.Clusters is not None:
            self.model.Clusters = [torch.as_tensor(c.astype(int)).squeeze().to(self.params_tr.device) for c in self.Clusters ]
        
        # Create training and validation loaders
        self.loader_tr = self.makeLoader(data_tr)
        self.loader_va = self.makeLoader(data_va)
        
        # Print training header
        self.params_tr.printHeader()
        
        # Reset the model parameters (to random)
        self.model.reset_parameters()
        
        # Reset the checkpoints
        for checkpoint in self.params_tr.checkpoints: checkpoint.reset()
        
        # Move the model to the device
        self.model = self.model.to(self.params_tr.device)
        
        # Create an optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params_tr.learning_rate)
        
        # Initialize results object
        results_tr = GN4IP.train.TrainingResults()
        results_tr.names.append(name)
        results_tr.loss_tr.append([])
        results_tr.loss_va.append([])
        
        # Prepare for main loop
        GN4IP.utils.printLine()
        tstart = time.time()
        
        # Main loop
        for epoch in range(1, self.params_tr.max_epochs+1):
        
            # Do the training and validation loops
            results_tr.loss_tr[0].append(self.trainEpoch())
            results_tr.loss_va[0].append(self.validEpoch())
            
            # Evaluate the checkpoints
            for checkpoint in self.params_tr.checkpoints: results_tr = checkpoint.evaluate(results_tr, self.model)
        
            # Check if need to stop
            if results_tr.stop: break
            
        # Record total training time and model name
        results_tr.time_tr.append(time.time() - tstart)
        
        # Get the stop_model parameters if EarlyStopping was used
        if results_tr.stop_model is not None:
            self.model.load_state_dict(results_tr.stop_model)
            results_tr.stop_model = None
        
        # Move back to cpu and save
        GN4IP.utils.printLine()
        self.model = self.model.to("cpu")
        torch.save(self.model.state_dict(), name)
        print("#  Saved model as:", name)
        GN4IP.utils.printLine()
        
        return results_tr
    
    def makeLoader(self, data):
        '''
        Create a loader for the data (typically training or validation data).
        The data is passed as an np.array in the format of 
        [[x1,...,xn,y], samples, nodes]. Use the self.params_tr.
        '''
        
        if self.model.type == "gnn":
            loader = self.makeLoaderGNN(data)
        
        elif self.model.type == "cnn2d" or self.model.type == "cnn3d":
            loader = self.makeLoaderCNN(data)
        
        return loader
        
    def makeLoaderGNN(self, data):
        '''
        Make a loader for a GNN.
        '''
        
        # Make sure the edges are a tensor
        edges = torch.as_tensor(self.Edges[0].astype(int))
        
        # Convert the data into tensors
        x = torch.as_tensor(np.array(data[0:-1]), dtype=torch.float64)
        y = torch.as_tensor(         data[  -1] , dtype=torch.float64)
        
        # Scale the tensors
        x = torch.mul(x, self.params_tr.scale)
        y = torch.mul(y, self.params_tr.scale)
        
        # Permute the order of axes to get [samples, nodes, features]
        x = x.permute(1,2,0)
        
        # Loop through the samples to make a dataset of graph objects
        dataset = []
        for i in range(x.size(0)):
            dataset.append(Data(edge_index=edges, x=x[i,:,:], y=y[i,:].unsqueeze(dim=1)))
        
        # Prepare loader and return
        loader = DataLoader(dataset, batch_size=self.params_tr.batch_size, shuffle=True)
        
        return loader
    
    def makeLoaderCNN(self, data):
        '''
        Make a loader for a CNN. Input data is on a graph so use the 
        self.params_tr.interpolator to move to a grid.
        '''
        
        # Use the interpolator (need data in [samples, features, nodes])
        mesh_data = np.array(data).transpose(1,0,2)
        grid_data = self.params_tr.interpolator.interpMesh2Grid(mesh_data)
        
        # Convert the grid_data into tensor and scale
        grid_data = torch.as_tensor(grid_data, dtype=torch.float64)
        grid_data = torch.mul(grid_data, self.params_tr.scale)
        
        # Make a dataset and a loader and return
        dataset = torch.utils.data.TensorDataset(grid_data[:,0:-1,:], grid_data[:,-1:,:])
        loader = DataLoader(dataset, batch_size=self.params_tr.batch_size, shuffle=True)
        return loader
    
    def trainEpoch(self):
        '''
        Loop through the batches in self.loader_tr. For each batch, evaluate 
        the samples by computing the loss. Use self.optimizer to update the 
        parameters after each batch. Then, return the average loss (per sample)
        '''
        
        # Put the model in training mode
        self.model.train()
        
        # Prepare loss counters
        N    = 0
        loss = 0
        
        # Loop through the batches of the training loader
        for batch in self.loader_tr:
            self.optimizer.zero_grad()
            loss_batch, n = self.evaluate_batch(batch)
            loss_batch.backward()
            self.optimizer.step()
            N    += n
            loss += n*loss_batch.item()
        
        # Return the average loss per sample
        return loss / N
    
    def validEpoch(self):
        '''
        Loop through the batches in self.loader_va. For each batch, evaluate 
        the samples by computing the loss. Then, return the average loss (per
        sample)
        '''
        
        # Put the model in evaluation model
        self.model.eval()
        
        # Prepare loss counter
        N    = 0
        loss = 0
        
        # Loop through the batches in the loader (without computing gradients)
        with torch.no_grad():
            for batch in self.loader_va:
                loss_batch, n = self.evaluate_batch(batch)
                N    += n
                loss += n*loss_batch.item()
        
        # Return the average loss per sample
        return loss / N
    
    def evaluate_batch(self, batch):
        '''
        Depending on the self.model.type, use the model to predict on the
        batch. Then, compare the predictions and the labels using the 
        self.model.loss_function. Return the loss value and number of samples
        in the batch.
        '''
        
        # If gnn, feed data into gnn (moving data to model's device)
        if self.model.type == "gnn":
            pred  = self.model(batch.x.to(self.params_tr.device), batch.edge_index.to(self.params_tr.device))
            label = batch.y.to(self.params_tr.device)
            n = batch.num_graphs
        
        # If cnn, feed data into cnn
        elif self.model.type == "cnn2d" or self.model.type == "cnn3d":
            pred  = self.model(batch[0].to(self.params_tr.device))
            label = batch[1].to(self.params_tr.device)
            n = batch[0].size(0)
        
        # Otherwise error
        else:
            raise ValueError("Wrong type in model. It must be gnn, cnn2d, or cnn3d.")
        
        # Compute the loss and return
        loss_batch = self.model.loss_function.evaluate(pred, label, n)
        return loss_batch, n
    
    def predictModel(self, data_pr):
        '''
        Use self.model and self.params_pr to make predictions for the data_pr.
        The procedure is different for GNNs and CNNs, but a PredictionResults
        object is returned either way.
        '''
        
        # Assign self.Edges and self.Clusters to self.model as tensors if not None
        if self.Edges is not None:
            self.model.Edges    = [torch.as_tensor(e.astype(int)).squeeze().to(self.params_pr.device) for e in self.Edges[1:]]
        if self.Clusters is not None:
            self.model.Clusters = [torch.as_tensor(c.astype(int)).squeeze().to(self.params_pr.device) for c in self.Clusters ]
        
        # Initialize results output with the network inputs
        results_pr = GN4IP.predict.PredictionResults(data_pr[0])
        
        # Move the model to the device
        self.model = self.model.to(self.params_pr.device)
        
        # Initialize a prediction tensor
        predictions = torch.zeros_like(torch.as_tensor(data_pr[0]), dtype=torch.float64)
        
        # Start a timer
        tstart = time.time()
        
        # For GNN models
        if self.model.type == "gnn":
            
            # Convert the data to a tensor and scale
            x = torch.as_tensor(np.array(data_pr), dtype=torch.float64)
            x = torch.mul(x, self.params_pr.scale)
            
            # Permute the input data [C,N,M] -> [N,M,C]
            x = x.permute(1,2,0)
            
            # Make sure the edges are a tensor
            edges = torch.as_tensor(self.Edges[0].astype(int)).to(self.params_pr.device)
            
            # Loop through the samples and predict
            with torch.no_grad():
                for i in range(x.size(0)):
                    predictions[i,:] = self.model(x[i,:].to(self.params_pr.device), edges).squeeze().cpu()
            
            # Make sure prediction arrays are numpy
            predictions = predictions.numpy()
            predictions_cnn = None
        
        else:
            
            # Embed the data to a pixel grid
            results_pr.input_cnn = None
            # Loop through the samples and predict
            predictions_cnn = None
            # Embed the prediction back to the graph
            prediction = None
        
        # Update the results to include the predictions and time
        results_pr.predictions.append(predictions)
        results_pr.predictions_cnn.append(predictions_cnn)
        results_pr.time_pr.append(time.time() - tstart)
        
        return results_pr
    
    def checkNames(self, OVERWRITE):
        '''
        Check if the files with names in self.model_names exist already. Print
        a summary that includes if they should be overwritten and return a list
        of booleans the same length as self.model_names that determines if the 
        models should be trained
        '''
        
        print(self.model_names)
        if OVERWRITE:
            check = []
            for name in self.model_names:
                check.append(True)
        print(check)
        print("Add an actual check to LR.checkNames()")
        return check