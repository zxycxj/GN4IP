# This is an example of using the interpolator

import GN4IP
import numpy as np
import scipy.io

# Load the dataset
dataset_name = "Chest_16"
filename = "/mmfs1/home/4106herzbew/thesis/on_github/GN4IP/datasets/" + dataset_name + ".mat"
keys = ["X2", "Y", "Edges", "Clusters", "Centers"]
[X2, Y, Edges, Clusters, Centers] = GN4IP.utils.loadDataset(filename, keys)
GN4IP.utils.printLine()
print("Loaded the following keys from the " + dataset_name + " dataset:")
print(keys)

# Reduce the number of samples
X2 = X2[0:4,:]

# Create an interpolator object
grid_size = 8
interpolator = GN4IP.utils.interpolator(Centers[0], grid_size)
print("Created interpolator with mesh points of size", Centers[0].shape, "and grid_size of", grid_size)


# Prepare data to interpolate: mesh_data ~ [N,C,M]
print(X2.shape)
mesh_data = np.expand_dims(X2, 1)
print(mesh_data.shape)
grid_data = interpolator.interpMesh2Grid(mesh_data)
print(grid_data.shape)
mesh_data_2 = interpolator.interpGrid2Mesh(grid_data)
print(mesh_data_2.shape)

# Save the data to examine in matlab
save_dict = {
    "mesh_data" : mesh_data,
    "grid_data" : grid_data,
    "mesh_data_2" : mesh_data_2,
    "mesh_points" : interpolator.mesh_points,
    "grid_points" : interpolator.grid_points
}
scipy.io.savemat("ex_interpolate_results.mat", save_dict)