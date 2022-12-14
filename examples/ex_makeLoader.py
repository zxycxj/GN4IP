# This is an example of using the LR.makeLoader method

import GN4IP


# Load a dataset
dataset_name = "Chest_16"
filename = "/mmfs1/home/4106herzbew/thesis/on_github/GN4IP/datasets/" + dataset_name + ".mat"
keys = ["X1", "X2", "Y", "Edges", "Centers"]
[X1, X2, Y, Edges, Centers] = GN4IP.utils.loadDataset(filename, keys)
GN4IP.utils.printLine()
print("Loaded the following keys from the " + dataset_name + " dataset:")
print(keys)



# Organize the data how it is usually used
data_1 = [X1, Y]
data_2 = [X1, X2, Y]



# Set some training parameters
params_tr = GN4IP.train.TrainingParameters(
    batch_size = 2,
)



# Create a GNN model with 1 input
model_gnn_1 = GN4IP.models.buildModel(
    type          = "gnn",
    channels_in   = 1
)

# Create a PostProcess object for that model
LR_gnn_1 = GN4IP.LearnedReconstruction(
    model_names = "gnn_1",
    model       = model_gnn_1
)
LR_gnn_1.params_tr = params_tr
LR_gnn_1.Edges = Edges

# Make a loader
loader_gnn_1 = LR_gnn_1.makeLoader(data_1)
print(vars(loader_gnn_1))



# Create a GNN model with 2 inputs
model_gnn_2 = GN4IP.models.buildModel(
    type          = "gnn",
    channels_in   = 2
)

# Create a PostProcess object for that model
LR_gnn_2 = GN4IP.LearnedReconstruction(
    model_names = "gnn_2",
    model       = model_gnn_2
)
LR_gnn_2.params_tr = params_tr
LR_gnn_2.Edges = Edges

# Make a loader
loader_gnn_2 = LR_gnn_1.makeLoader(data_2)
print(vars(loader_gnn_2))



# Create an interpolator object
grid_size = 8
interpolator = GN4IP.utils.interpolator(Centers[0], grid_size)
params_tr.interpolator = interpolator
print("Created interpolator with mesh points of size", Centers[0].shape, "and grid_size of", grid_size)



# Create a CNN2d model with 1 input
model_cnn2d_1 = GN4IP.models.buildModel(
    type          = "cnn2d",
    channels_in   = 1
)

# Create a PostProcess object for that model
LR_cnn2d_1 = GN4IP.LearnedReconstruction(
    model_names = "cnn2d_1",
    model       = model_cnn2d_1
)
LR_cnn2d_1.params_tr = params_tr

# Make a loader
loader_cnn2d_1 = LR_cnn2d_1.makeLoader(data_1)
print(vars(loader_cnn2d_1))



# Create a CNN2d model with 2 inputs
model_cnn2d_2 = GN4IP.models.buildModel(
    type          = "cnn2d",
    channels_in   = 2
)

# Create a PostProcess object for that model
LR_cnn2d_2 = GN4IP.LearnedReconstruction(
    model_names = "cnn2d_2",
    model       = model_cnn2d_2
)
LR_cnn2d_2.params_tr = params_tr

# Make a loader
loader_cnn2d_2 = LR_cnn2d_2.makeLoader(data_2)
print(vars(loader_cnn2d_2))


