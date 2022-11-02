# This is an example of using the ModelBased object to train a series of models

import GN4IP


# Load the dataset
dataset_name = "Chest_16"
filename = "/mmfs1/home/4106herzbew/thesis/on_github/GN4IP/datasets/" + dataset_name + ".mat"
keys = ["X1", "Y", "Edges", "Clusters"]
[X1, Y, Edges, Clusters] = GN4IP.utils.loadDataset(filename, keys)
GN4IP.utils.printLine()
print("Loaded the following keys from the " + dataset_name + " dataset:")
print(keys)
X1 = X1[0:4,:]
Y  =  Y[0:4,:]

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


# Get training parameters ready
params_tr = GN4IP.train.TrainingParameters(
    device = "cpu",
    batch_size = 2,
    max_epochs = 2,
    checkpoints = [
        GN4IP.train.PrintLoss(),
        GN4IP.train.EarlyStopping(pat_reset=3)
    ],
    updateFunction = dummyUpdate,
    fwd_data_file = ""
)


# Train the models in the ModelBased object
mb1.train(
    data_tr = [X1, None, Y], # [Input1, input2, output]
    data_va = [X1, None, Y],
    params_tr = params_tr,
    Edges = Edges,
    Clusters = Clusters,
    OVERWRITE = True
)
print(vars(mb1.results_tr))
print("Done Training!")


# Save the training results
results_tr_name = model_dir + "ModelBased_test1.mat"
mb1.results_tr.save(results_tr_name)