# Functions for training models

import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from GN4IP.utils.message import printLine
#from GN4IP.train.checkpoints import evaluateCheckpoints


# Train a model
def trainModel(device, loaders, model, Edges, Clusters, learning_rate, max_epochs, checkpoints, model_num=-1):
    
    # Reset the parameters of the model
    model.reset_parameters()
    
    # Make a deep copy of the checkpoints list so it doesn't get changed outside this function
    #checkpoints = copy.deepcopy(checkpoints)
    for checkpoint in checkpoints: checkpoint.reset()
    
    # Move the model to the device and add edges and clusters to it if gnn
    model          = model.to(device)
    if model.type == "gnn":
        model.Edges    = [torch.as_tensor(e.astype(int)-1).squeeze().to(device) for e in Edges[1:]]
        model.Clusters = [torch.as_tensor(c.astype(int)-1).squeeze().to(device) for c in Clusters ]
    
    # Create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize training lists
    loss_tr = []
    loss_va = []
    
    # Start the main loop
    printLine()
    stop = False
    tstart = time.time()
    for epoch in range(1, max_epochs+1):
        
        # Do the training and validation loops
        loss_tr.append(train(model, loaders[0], optimizer))
        loss_va.append(valid(model, loaders[1]))
    
        # Go through the checkpoints and evaluate
        #checkpoints = evaluateCheckpoints(checkpoints, model, loaders[1], loss_tr, loss_va, model_num)
        for checkpoint in checkpoints: checkpoint.evaluate(model, loaders[1], loss_tr, loss_va, model_num)
        
        # If earlyStopping checkpoint is used, might need to stop early
        for checkpoint in checkpoints:
            if checkpoint.name == "earlyStopping":
                if checkpoint.current_pat <= 0 and len(loss_tr) >= checkpoint.min_epochs:
                    if checkpoint.params2keep == "best":
                        if checkpoint.other_name:
                            saveModel(model, checkpoint.other_name)
                        model.load_state_dict(checkpoint.best_model)
                        print("#  Training stopped early on epoch "+str(checkpoint.best_epoch)+". Loading best params.")
                    elif checkpoint.params2keep == "last":
                        print("#  Training stopped early on epoch "+str(len(loss_tr))+". Keeping last parameters.")
                        if checkpoint.other_name:
                            model2 = copy.deepcopy(model)
                            model2.load_state_dict(checkpoint.best_model)
                            saveModel(model2, checkpoint.other_name)
                    stop = True
        if stop:
            break
    
    # Done training
    printLine()
    
    # Check the savePredictions checkpoint for best-model-prediction save
    for checkpoint in checkpoints:
        if checkpoint.name == "savePredictionCheckpoint":
            if checkpoint.frequency == 1e7:
                checkpoint.frequency = -1
                checkpoint.evaluate(model, loaders[1], loss_tr, loss_va, model_num)
    
    # Determine training time
    time_tr = time.time() - tstart
    return model, loss_tr, loss_va, time_tr

# training loop
def train(model, loader, optimizer):

    # Put the model in training mode
    model.train()
    
    # Prepare loss counter
    N    = 0
    loss = 0
    
    # Loop through the batches in the loader
    for batch in loader:
        optimizer.zero_grad()
        loss_batch, n = eval_batch(model, batch)
        loss_batch.backward()
        optimizer.step()
        N    += n
        loss += n*loss_batch.item()
    
    # Return the average loss per sample
    return loss / N
    
# validation loop
def valid(model, loader):
    
    # Put the model in evaluation model
    model.eval()
    
    # Prepare loss counter
    N    = 0
    loss = 0
    
    # Loop through the batches in the loader (without computing gradients)
    with torch.no_grad():
        for batch in loader:
            loss_batch, n = eval_batch(model, batch)
            N    += n
            loss += n*loss_batch.item()
    
    # Return the average loss per sample
    return loss / N

# evaluate a batch
def eval_batch(model, batch):
    
    # If gnn, feed data into gnn (moving data to model's device)
    if model.type == "gnn":
        pred  = model(batch.x.to(model.get_device()), batch.edge_index.to(model.get_device()))
        label = (batch.y-batch.x[:,0:1] if model.res else batch.y).to(model.get_device())
        n = batch.num_graphs
    
    # If cnn, feed data into cnn
    elif model.type == "cnn2d" or model.type == "cnn3d":
        pred  = model(batch[0].to(model.get_device()))
        label = (batch[1]-batch[0][:,0:1] if model.res else batch[1]).to(model.get_device())
        n = batch[0].size(0)
    
    # Otherwise error
    else:
        raise ValueError("Wrong type in train(). It must be gnn, cnn2d, or cnn3d.")
    
    # Compute the loss and return
    #loss_batch = model.loss_function(pred, label)
    loss_batch = model.loss_function.evaluate(pred, label, n)
    return loss_batch, n
