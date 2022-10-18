# This is an example of building a graph U-net

import GN4IP

model = GN4IP.model.build(
    type          = "gnn",
    device        = "cpu",
    depth         = 1,
    channels_in   = 1,
    channels      = 8,
    loss_function = torch.nn.MSELoss()
)
GN4IP.model.parameterSummary(model)