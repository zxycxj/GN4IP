# Describe a class used for model-based learned reconstruction

import time
import torch
import GN4IP
from os import path

class ModelBased(object):
    '''
    A ModelBased object contains a model and a list of model names. During
    training, the model parameters can be loaded from file or trained and saved
    at each iteration. During prediction, the model parameters will be loaded 
    from file at each iteration. Always check, before doing anything, if the
    files for the model parameters exist already. They can be overwritten
    during training if specified.
    '''
    
    def __init__(self, model_names, model=None):
        '''
        A ModelBased object is initialized with just a list of model names. The
        number of names in the list will determine how many iterations of 
        networks are trained/used for prediction. A model itself will usually 
        be specified too unless the default model architecutre is being used. 
        '''
        self.model_names = model_names
        self.model = model
        self.params_tr = None
        self.results_tr = None
        self.params_pr = None
        self.results_pr = None
    
    def train(self, data_tr, data_va, params_tr, Edges=None, Clusters=None, OVERWRITE=False):
        '''
        For each name in model_names, the parameters in self.model will be 
        trained using data_tr, data_va, and params_tr. If self.model is None,
        it will be built beforehand. After training the parameters at each
        iteration, they are saved using model_names. At each iteration, if 
        model_names[i] already exists, overwrite is False, and the shape of the
        weights match, the weights won't be trained and overwritten. Instead
        they will be loaded and used to get the next iterate
        '''
        
        # Add the params_tr, Edges, and Clusters to self
        self.params_tr = params_tr
        self.params_pr = params_tr # need params_pr.scale = params_tr.scale
        self.Edges = Edges
        self.Clusters = Clusters
        self.results_tr = GN4IP.train.TrainingResults()
        
        # If model is None, build it
        if self.model is None:
            self.model = GN4IP.models.build()
        
        # Assign the Edges and Clusters to the model as tensors if not None
        if self.model.type == "gnn":
            if self.Edges is not None:
                self.model.Edges    = [torch.as_tensor(e.astype(int)).squeeze().to(self.params_tr.device) for e in self.Edges[1:]]
            if self.Clusters is not None:
                self.model.Clusters = [torch.as_tensor(c.astype(int)).squeeze().to(self.params_tr.device) for c in self.Clusters ]
        
        # Check which models can be/will be loaded
        self.checkNames(OVERWRITE)
        
        # For each name in model_names
        for name in self.model_names:
            
            # Compute second input term using the update function
            if data_tr[1] is None:
                data_tr[1] = self.params_tr.updateFunction(data_tr[0], self.params_tr.fwd_data)
            if data_va[1] is None:
                data_va[1] = self.params_tr.updateFunction(data_va[0], self.params_tr.fwd_data)
            
            # If the file_name already exists and we choose not to overwrite
            if (path.exists(name) and not OVERWRITE):
            
                # Load the model weights (error might raise if state_dict sizes don't match?)
                self.model.load_state_dict(torch.load(name))
                self.results_tr.loadedModel(name) # Add placeholder to self.results_tr for this iteration
                
            # Otherwise
            else:
            
                # Train the model weights using the data
                self.results_tr.appendResults(self.trainModel(data_tr, data_va, name))
                
                # Save the model parameters in self.results_tr.stop_model
                GN4IP.utils.saveModel(self.model, name)
                
            # Use the model to compute the next iterates
            data_tr[0] = self.predictModel(data_tr[0:-1]).predictions # Update this based on what results_pr looks like
            data_va[0] = self.predictModel(data_va[0:-1]).predictions
            data_tr[1] = None
            data_va[1] = None
    
    def checkNames(self, OVERWRITE):
        '''
        Check if the names in model_names exist and if the weights in those
        files can be loaded into self.model. If OVERWRITE is True, it doesn't
        matter if the files are there but it is still nice to know that files
        will be written over.
        '''
        
        GN4IP.utils.printLine()
        print("#  Checking the model names...")
        GN4IP.utils.printLine()
        print("#  Num | Exist | Fits  | Write | Name")
        for i in range(self.model_names):
            if path.exists(name):
                try:
                    self.model.load_state_dict(torch.load(self.model_names[i]))
                    print("#  "+str(i)+ " | True  | True  |", OVERWRITE, "| "+self.model_names[i])
                except:
                    print("#  "+str(i)+" | True  | False | True  | "+self.model_names[i])
            else:
                print("#  "+str(i)+" | False |       | True  | "+self.model_names[i])
    
    def trainModel(self, data_tr, data_va, name):
        '''
        Train self.model with the training and validation data provided and
        according to the self.params_tr. The data is passed as graph data. 
        Return a training Results object that includes the loss values and 
        training time.
        '''
        
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
        
        # Initialize results
        results_tr = GN4IP.train.trainingResults()
        
        # Prepare for main loop
        GN4IP.utils.printLine()
        self.stop = False
        tstart = time.time()
        
        # Main loop
        for epoch in range(1, self.params_tr.max_epochs+1):
        
            # Do the training and validation loops
            results_tr.loss_tr[-1].append(self.trainEpoch())
            results_tr.loss_va[-1].append(self.validEpoch())
            
            # Evaluate the checkpoints
            for checkpoint in self.params_tr.checkpoints: self = checkpoint.evaluate(self)
        
            # Check if need to stop
            if self.stop: break
            
        # Record total training time
        results_tr.time_tr[-1] = time.time() - tstart
        
        # Move the model back to cpu
        self.model = self.model.to("cpu")
        
        return results_tr
    
    def makeLoader(self, data):
        '''
        Create a loader for the data. The data is a list where the last item is
        the ground truth and the prior items are the inputs. Loaders are really
        only needed for training so they use self.params_tr.
        '''
        
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
    
    def makeLoaderGNN(self, x, y):
        '''
        Finish the loader for GNN data
        '''
        
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
        '''
        Finish the loader for (2D) CNN data
        '''
        return 1
    
    # Finish making the (3D) CNN Loader  
    def makeLoaderCNN3d(self, x, y):
        '''
        Finish the loader for (3D) CNN data
        '''
        return 1
    
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
        self.checkNames(None)
        
        # Initialize the prediction results
        self.results_pr = GN4IP.predictions.PredictionResults(data_pr[0], data_pr[1])
        
        # For each name in model_names
        for name in self.model_names:
            
            # Make sure the data is all there
            if data_pr[1] is None:
                data_pr[1] = self.params_pr.updateFunction(data_pr[0], self.params_pr.fwd_data)
            
            # Load the model parameters
            self.model.load_state_dict(torch.load(name))
            
            # Use the model to compute the next iterates
            self.results_pr.appendPredictions(self.predictModel(data_pr))
            
            # Prepare for the next iteration
            data_pr[0] = self.results_pr.predictions[-1]
            data_pr[1] = None
    
    def predictModel(self, data_pr):
        '''
        Use self.model and self.params_pr to make predictions for the data_pr.
        The procedure is different for GNNs and CNNs, but a PredictionResults
        object is returned either way
        '''
        
        # Move the model to the device
        self.model = self.model.to(self.params_pr.device)
        
        # Convert the data to a tensor and scale
        x = torch.as_tensor(np.array(data_pr), dtype=torch.float64)
        x = torch.mul(x, self.params_pr.scale)
        
        # Initialize a prediction tensor
        predictions = torch.zeros_like(torch.as_tensor(data_pr[0]), dtype=torch.float64)
        
        # Start a timer
        tstart = time.time()
        
        # For GNN models
        if self.model.type == "gnn":
            
            # Permute the input data [C,N,M] -> [N,C,M]
            x = x.permute(1,0,2)
            
            # Make sure the edges are a tensor
            edges = torch.as_tensor(self.Edges[0].astype(int)).to(self.params_pr.device)
            
            # Loop through the samples and predict
            with torch.no_grad():
                for i in range(x.size(1)):
                    predictions[i,:] = self.model(x[i,:].to(self.params_pr.device), edges).squeeze().cpu()
            
            # There were no prediction results
            predictions_cnn = None
            
        else:
            
            # Embed the data to a pixel grid
            
            # Loop through the samples and predict
            
            # Embed the prediction back to the graph
        
        # Record the time and create the result output
        time_pr = time.time() - tstart
        results_pr = GN4IP.predict.PredictionResults(predictions, predictions_cnn, time_pr)
        
        return results_pr
    
    
    
    
    
    
    def savePredictions(self, results_name):
        '''
        If self.results_pr is not None, then the results are saved in a .mat 
        file with the name given by results_name
        '''
        
        if self.results_pr is not None:
            self.results_name = results_name
            GN4IP.utils.savePredictions(results_pr, results_name)
        else:
            ValueError("Unet attribute results_pr is None so nothing can be saved!")