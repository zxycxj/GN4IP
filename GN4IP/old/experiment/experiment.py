# This script defines an experiment class that can be used to train and predict a model

import time
import torch
import GN4IP
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class experiment(object):

    def __init__(self, device="cpu", model=None, data_tr=None, data_va=None, params_tr=None, Edges=None, Centers=None, Clusters=None):
        self.device = device
        self.model = model
        self.data_tr = data_tr
        self.data_va = data_va
        self.params_tr = params_tr
        self.Edges = Edges
        self.Centers = Centers
        self.Clusters = Clusters
        
    # A method for training the model using the training data
    def trainUnet(self):
        
        # Assign the Edges and Clusters to the model as tensors if not None
        if self.model.type == "gnn":
            if self.Edges is not None:
                self.model.Edges    = [torch.as_tensor(e.astype(int)).squeeze().to(self.device) for e in self.Edges[1:]]
            if self.Clusters is not None:
                self.model.Clusters = [torch.as_tensor(c.astype(int)).squeeze().to(self.device) for c in self.Clusters ]
        
        # Create training and validation loaders
        self.loader_tr = self.makeLoader("tr")
        self.loader_va = self.makeLoader("va")
        
        # Print training header
        self.params_tr.printHeader()
        
        # Traing the model. Results stored in self.results_tr
        self.trainModel()
    
    # Create a loader for loading training/validation data into the model
    def makeLoader(self, type):
        
        # Check if training or validation
        if type == "tr":
            data = self.data_tr
        elif type == "va":
            data = self.data_va
        else:
            raise ValueError("The experiment.loader(type) method requires type to be 'tr' or 'va'")
        
        # Convert the data into tensors
        x = torch.as_tensor(np.array(data[0:-1]), dtype=torch.float64)
        y = torch.as_tensor(         data[  -1] , dtype=torch.float64)
        
        # Scale the tensors
        x = torch.mul(x, self.params_tr.scale)
        y = torch.mul(y, self.params_tr.scale)
        
        # Make the loader for a GNN
        if self.model.type == "gnn":
            loader = self.makeLoaderGNN(x, y)
        
        # Make the loader for a CNN (2d)
        elif self.model.type == "cnn2d":
            loader = self.makeLoaderCNN2d(x, y)
        
        # Make the loader for a CNN (3d)
        elif self.model.type == "cnn3d":
            loader = self.makeLoaderCNN3d(x, y)
        
        # Return the loader
        return loader
    
    # Finish making the GNN Loader
    def makeLoaderGNN(self, x, y):
        
        # Make sure the edges are a tensor
        edges = torch.as_tensor(self.Edges[0].astype(int))
        
        # Permute the order of axes to get [samples, nodes, features]
        x = x.permute(1,2,0)
        
        # Loop through the samples to make a dataset of graph objects
        dataset = []
        for i in range(x.size(0)):
            dataset.append(Data(edge_index=edges, x=x[i,:,:], y=y[i,:].unsqueeze(dim=1)))
        
        # Prepare loader and return
        loader = DataLoader(dataset, batch_size=self.params_tr.batch_size, shuffle=True)
        return loader
    
    # Finish making the (2D) CNN Loader
    def makeLoaderCNN2d(self, x, y):
        return 1
    
    # Finish making the (3D) CNN Loader  
    def makeLoaderCNN3d(self, x, y):
        return 1
    
    # Train a model
    def trainModel(self):
        
        # Reset the model parameters
        self.model.reset_parameters()
        
        # Reset the checkpoint
        for checkpoint in self.params_tr.checkpoints: checkpoint.reset()
        
        # Move the model to the device and add edges
        self.model = self.model.to(self.device)
        
        # Create an optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params_tr.learning_rate)
        
        # Initialize results
        self.results_tr = GN4IP.experiment.trainingResults()
        
        # Prepare for main loop
        GN4IP.utils.printLine()
        self.stop = False
        tstart = time.time()
        
        # Main loop
        for epoch in range(1, self.params_tr.max_epochs+1):
        
            # Do the training and validation loops
            self.results_tr.loss_tr.append(self.train())
            self.results_tr.loss_va.append(self.valid())
            
            # Evaluate the checkpoints
            for checkpoint in self.params_tr.checkpoints: self = checkpoint.evaluate(self)
        
            # Check if need to stop
            if self.stop: break
            
        # Record total training time
        self.results_tr.time_tr = time.time() - tstart
        
        # Move the model back to cpu
        self.model = self.model.to("cpu")
    
    # The training loop
    def train(self):
        
        # Put the model in training mode
        self.model.train()
        
        # Prepare loss counters
        N    = 0
        loss = 0
        
        # Loop through the batches of the training loader
        for batch in self.loader_tr:
            self.optimizer.zero_grad()
            loss_batch, n = self.eval_batch(batch)
            loss_batch.backward()
            self.optimizer.step()
            N    += n
            loss += n*loss_batch.item()
        
        # Return the average loss per sample
        return loss / N
    
    # The validation loop
    def valid(self):
    
        # Put the model in evaluation model
        self.model.eval()
        
        # Prepare loss counter
        N    = 0
        loss = 0
        
        # Loop through the batches in the loader (without computing gradients)
        with torch.no_grad():
            for batch in self.loader_va:
                loss_batch, n = self.eval_batch(batch)
                N    += n
                loss += n*loss_batch.item()
        
        # Return the average loss per sample
        return loss / N
    
    # Evaluate a batch
    def eval_batch(self, batch):
        
        # If gnn, feed data into gnn (moving data to model's device)
        if self.model.type == "gnn":
            pred  = self.model(batch.x.to(self.device), batch.edge_index.to(self.device))
            label = batch.y.to(self.device)
            n = batch.num_graphs
        
        # If cnn, feed data into cnn
        elif self.model.type == "cnn2d" or self.model.type == "cnn3d":
            pred  = self.model(batch[0].to(self.device))
            label = batch[1].to(self.device)
            n = batch[0].size(0)
        
        # Otherwise error
        else:
            raise ValueError("Wrong type in model. It must be gnn, cnn2d, or cnn3d.")
        
        # Compute the loss and return
        loss_batch = self.model.loss_function.evaluate(pred, label, n)
        return loss_batch, n
