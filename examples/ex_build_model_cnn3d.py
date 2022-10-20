# This is an example of building a 3D CNN U-net

import time
import GN4IP
import torch

tstart = time.time()
model = GN4IP.models.buildModel(
    type          = "cnn3d",
    depth         = 1,
    channels_in   = 1,
    channels      = 8,
    loss_function = torch.nn.MSELoss()
)
GN4IP.models.parameterSummary(model)
GN4IP.utils.printLine()
GN4IP.utils.timeMessage(tstart, "Built a (3D) CNN!")