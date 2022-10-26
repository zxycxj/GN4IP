# This is an example of training a GNN using the experiment class

import time
import GN4IP
import torch

# Start a timer
tstart = time.time()

# Load a dataset
filename = "/mmfs1/home/4106herzbew/thesis/on_github/GN4IP/datasets/Chest_16.mat"
keys = ["X3", "Y", "Edges", "Clusters"]
[X3, Y, Edges, Clusters] = GN4IP.utils.loadDataset(filename, keys)

# Split data into training and validation
N_tr = 6
N_va = 3 + N_tr
data_tr = [X3[    :N_tr, :],  Y[    :N_tr, :]]
data_va = [X3[N_tr:N_va, :],  Y[N_tr:N_va, :]]

# Build a model
model = GN4IP.models.buildModel(
    type          = "gnn",
    depth         = 1,
    channels_in   = 1,
    channels      = 8,
    loss_function = GN4IP.utils.myMSELoss()
)
GN4IP.utils.printLine()
GN4IP.utils.timeMessage(tstart, "Built a GNN!")

# Prepare the training parameters
params_tr = GN4IP.experiment.trainingParameters(
    scale = 1,
    batch_size = 2,
    learning_rate = 0.005,
    device = "cpu",
    max_epochs = 50,
    checkpoints = [
        GN4IP.experiment.printLoss(),
        GN4IP.experiment.earlyStopping(pat_reset=4)
    ]
)

# Create an experiment
ex1 = GN4IP.experiment.experiment(
    model = model,
    data_tr = data_tr,
    data_va = data_va,
    params_tr = params_tr,
    Edges = Edges,
    Clusters = Clusters,
)

# Train the model in the experiment
ex1.trainUnet()

# Print a summary of the training
ex1.results_tr.printSummary()

# Check the training time and loss lists
GN4IP.utils.printLine()
GN4IP.utils.timeMessage(tstart, "Complete")