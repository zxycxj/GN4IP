# This is an example of using the PostProcess object to predict on new samples

import GN4IP


# Load the dataset
dataset_name = "Chest_15"
filename = "/mmfs1/home/4106herzbew/thesis/on_github/GN4IP/datasets/" + dataset_name + ".mat"
keys = ["X2", "Y", "Edges", "Clusters"]
[X2, Y, Edges, Clusters] = GN4IP.utils.loadDataset(filename, keys)
GN4IP.utils.printLine()
print("Loaded the following keys from the " + dataset_name + " dataset:")
print(keys)
# Reduce the number of samples
X2 = X2[0:3,:]


# Create a model to use (Same architecture as the one that was saved)
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


# Get prediction parameters ready
params_pr = GN4IP.predict.PredictionParameters(
    device = "cpu"
)


# Predict with the model in the PostProcess object
unet1.predict(
    data_pr = [X2],
    params_pr = params_pr,
    Edges = Edges,
    Clusters = Clusters
)
print(vars(unet1.results_pr))
print("Done Predicting!")