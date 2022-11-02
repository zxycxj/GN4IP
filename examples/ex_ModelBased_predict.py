# This is an example of using the ModelBased object to train a series of models

import GN4IP


# Load the dataset
dataset_name = "Chest_15"
filename = "/mmfs1/home/4106herzbew/thesis/on_github/GN4IP/datasets/" + dataset_name + ".mat"
keys = ["X1", "Y", "Edges", "Clusters"]
[X1, Y, Edges, Clusters] = GN4IP.utils.loadDataset(filename, keys)
GN4IP.utils.printLine()
print("Loaded the following keys from the " + dataset_name + " dataset:")
print(keys)
X1 = X1[0:4,:]


# Create a model to use
model = GN4IP.models.buildModel(
    type          = "gnn",
    depth         = 2,
    channels_in   = 2, # Need/want two channels for ModelBased objects
    channels      = 8,
)


# Create a ModelBased object with the model
model_dir = "/mmfs1/home/4106herzbew/thesis/on_github/GN4IP/trained_networks/"
names_list = ["ModelBased_test1_1", "ModelBased_test1_2"]
model_names = [model_dir+name+".pt" for name in names_list]
mb1 = GN4IP.ModelBased(model_names, model)
GN4IP.models.parameterSummary(mb1.model)
GN4IP.utils.printLine()


# Define a dummy function for computing updates
def dummyUpdate(data, fwd_data_file):
    print("#  Computing updates...")
    GN4IP.utils.printLine()
    return data


# Get prediction parameters ready
params_pr = GN4IP.predict.PredictionParameters(
    device = "cpu",
    updateFunction = dummyUpdate,
    fwd_data_file = ""
)


# Predict with the models in the ModelBased object
mb1.predict(
    data_pr = [X1, None], # [Input1, input2]
    params_pr = params_pr,
    Edges = Edges,
    Clusters = Clusters
)
print(vars(mb1.results_pr))
print("Done predicting!")


# Save the prediction results
prediction_dir = "/mmfs1/home/4106herzbew/thesis/on_github/GN4IP/network_outputs/"
results_pr_name = prediction_dir + "ModelBased_test1_Chest15.mat"
mb1.results_pr.save(results_pr_name)