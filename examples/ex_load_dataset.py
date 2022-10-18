# This is an example of loading a dataset

import GN4IP
import time

# Start a timer
tstart = time.time()

# Pick a dataset
filename = "/mmfs1/home/4106herzbew/thesis/on_github/GN4IP/datasets/Chest_16.mat"
keys = ["X0", "X1", "Edges"]

# Print a message
GN4IP.utils.timeMessage(tstart, "Loading dataset "+filename)

# Load the dataset
[X0, X1, Edges] = GN4IP.utils.loadDataset(filename, keys)

# Print the things
print(X0)
print(X1)
print(Edges)
