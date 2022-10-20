# Function for training GCNM

import numpy as np
import torch
import GN4IP
from GN4IP.utils.message     import printLine
from GN4IP.utils.loaders     import createLoader
from GN4IP.train.train       import trainModel
from GN4IP.matlab.transform  import interp

# Train u-net model
def trainUNET(device, model, data_tr, data_va, Edges=[0], Clusters=[], embed_args=[], scale=1, batch_size=1, learning_rate=0.0001, max_epochs=1, checkpoints=[]):
    """
    Trains a u-net model using the data provided
    
    Args:
    
    
    
    """
    
    # Prepare output lists
    trained_models = []
    Loss_tr = []
    Loss_va = []
    Time_tr = []
    
    # If the models are cnns, embed the graph data to a pixel grid
    if model.type == "cnn2d" or model.type == "cnn3d":
        printLine()
        x = np.concatenate((data_tr+data_va), axis=0)
        N1 = data_tr[0].shape[0]
        N2 = data_va[0].shape[0]
        x_grid = interp(x, embed_args[0], embed_args[1], embed_args[2], embed_args[3], embed_args[4], embed_args[5])
        data_tr = [x_grid[    :  N1   , :], x_grid[  N1   :2*N1, :]]
        data_va = [x_grid[2*N1:2*N1+N2, :], x_grid[2*N1+N2:    , :]]
    
    # Create loaders
    loader_tr = createLoader(model.type, data_tr, scale, Edges[0], batch_size)
    loader_va = createLoader(model.type, data_va, scale, Edges[0], 1)
    
    # Train the model
    printLine()
    print("#  Training model")
    printLine()
    print("#   Training On:", device)
    print("#    Batch Size:", batch_size)
    print("# Learning Rate:", learning_rate)
    print("#         Scale:", scale)
    trained_model, loss_tr, loss_va, time_tr = trainModel(
        device        = device,
        loaders       = [loader_tr, loader_va],
        model         = model,
        Edges         = Edges,
        Clusters      = Clusters,
        learning_rate = learning_rate,
        max_epochs    = max_epochs,
        checkpoints   = checkpoints
    )
    trained_models.append(trained_model)
    Loss_tr.append(loss_tr)
    Loss_va.append(loss_va)
    Time_tr.append(time_tr)
    
    # Return the trained models and training information
    return trained_models, Loss_tr, Loss_va, Time_tr