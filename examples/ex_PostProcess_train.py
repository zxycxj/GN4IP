# This is an example of using the PostProcess object to train a model

import GN4IP


# Load the dataset
dataset_name = "Chest_16"
filename = "/mmfs1/home/4106herzbew/thesis/on_github/GN4IP/datasets/" + dataset_name + ".mat"
keys = ["X2", "Y", "Edges", "Clusters"]
[X2, Y, Edges, Clusters] = GN4IP.utils.loadDataset(filename, keys)
GN4IP.utils.printLine()
print("Loaded the following keys from the " + dataset_name + " dataset:")
print(keys)


# Create a model to use
model = GN4IP.models.buildModel(
    type          = "gnn",
    depth         = 2,
    channels_in   = 1,
    channels      = 8,
)


# Create a PostProcess object with the model
model_dir = "/mmfs1/home/4106herzbew/thesis/on_github/GN4IP/trained_networks/"
names_list = ["PostProcess_test1"]
model_names = [model_dir+name+".pt" for name in names_list]
unet1 = GN4IP.PostProcess(model_names, model)
GN4IP.models.parameterSummary(unet1.model)
GN4IP.utils.printLine()


# Get training parameters ready
params_tr = GN4IP.train.TrainingParameters(
    device = "cpu",
    batch_size = 5,
    max_epochs = 4,
    checkpoints = [
        GN4IP.train.PrintLoss(),
        GN4IP.train.EarlyStopping(pat_reset=3)
    ]
)


# Train the model in the PostProcess object
unet1.train(
    data_tr = [X2, Y],
    data_va = [X2, Y],
    params_tr = params_tr,
    Edges = Edges,
    Clusters = Clusters,
    OVERWRITE = True
)
print(vars(unet1.results_tr))
print("Done Training!")
