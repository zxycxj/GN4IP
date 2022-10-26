# Functions for creating models

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, max_pool_x
from tabulate import tabulate

from GN4IP.utils.message import printLine

# build a model
def buildModel(type, device="cpu", res=False, depth=0, channels_in=1, channels=8, convolutions=1, loss_function=torch.nn.MSELoss(), activation=F.relu, threshold=False, k_size=3, last_1x1=True, knn_up=False):
    
    # Make a gnn model
    if type == "gnn":
        model = gnnModel(res, depth, channels_in, channels, convolutions, loss_function, activation, threshold, last_1x1)
    
    # Make a cnn model (2d)
    elif type == "cnn2d":
        model = cnnModel2d(res, depth, channels_in, channels, convolutions, loss_function, activation, threshold, k_size, last_1x1, knn_up)
    
    # Make a cnn model (3d)
    elif type == "cnn3d":
        model = cnnModel3d(res, depth, channels_in, channels, convolutions, loss_function, activation, threshold, k_size, last_1x1, knn_up)
    
    # It might have some other type which is wrong
    else:
        raise ValueError("Wrong type in buildModel(). It must be gnn, cnn2d, or cnn3d.")
    
    # Make double and move to device before returning
    model = model.double()
    model.to(device)
    return model

# Make a class for gnn models
class gnnModel(torch.nn.Module):
    
    def __init__(self, res, depth, channels_in, channels, convolutions, loss_function, activation, threshold, last_1x1):
        super(gnnModel, self).__init__()
        self.type          = "gnn"
        self.res           = res
        self.depth         = depth
        self.channels_in   = channels_in
        self.channels      = channels
        self.convolutions  = convolutions
        self.loss_function = loss_function
        self.activation    = activation
        self.threshold     = threshold
        self.last_1x1      = last_1x1
        
        # Create initial convolutions
        chan_in  = channels_in
        chan_out = channels
        self.first_convs = self.buildGCNBlock(chan_in, chan_out)
        
        # Create the encoder side convolutions
        self.down_convs = torch.nn.ModuleList()
        for i in range(depth):
            chan_in  = chan_out
            chan_out = chan_out*2
            self.down_convs.append(self.buildGCNBlock(chan_in, chan_out))
        
        # Create the decoder side convolutions
        self.up_convs = torch.nn.ModuleList()
        for i in range(depth):
            chan_in  = chan_out
            chan_out = chan_out//2
            self.up_convs.append(self.buildGCNBlock(chan_in+chan_out, chan_out))
        
        # Create the final convolutions
        chan_in  = channels
        chan_out = 1
        self.final_conv = GCNConv(chan_in, chan_out)
        
        # Make sure to reset the parameters
        self.reset_parameters()
    
    # Use a function to add GCN blocks
    def buildGCNBlock(self, chan_in, chan_out):
        
        # Start a module list, build the block, and return it
        block = torch.nn.ModuleList()
        for i in range(self.convolutions):
            block.append(GCNConv(chan_in, chan_out))
            chan_in = chan_out
        return block
    
    # Use a function to reset all the model parameters
    def reset_parameters(self):
        
        # Loop through all convolution module lists and reset
        for conv in self.first_convs:
            conv.reset_parameters()
        for list in self.down_convs:
            for conv in list:
                conv.reset_parameters()
        for list in self.up_convs:
            for conv in list:
                conv.reset_parameters()
        self.final_conv.reset_parameters()
    
    # Define the forward pass
    def forward(self, x, edge_index, batch=None):
        
        # build the batch array
        if batch is None:
            batch = self.buildBatch(x, edge_index)
        
        # Initialize lists for storing skip connection info
        xs = []
        ei = []
        cl = []
        
        # Do the first convolutions
        x = self.doGCNBlock(x, edge_index, self.first_convs)
        
        # Do the encoder side with pooling and convolutions
        for i in range(self.depth):
            
            # Pooling
            xs += [x]
            ei += [edge_index]
            cl += [self.checkCluster(self.Clusters[i], batch)]
            x, batch = max_pool_x(cl[i], x, batch)
            edge_index = self.Edges[i]
            
            # Convolutions
            x = self.doGCNBlock(x, edge_index, self.down_convs[i])
        
        # Do the decoder side with upsampling and convolutions
        for i in range(self.depth):
            j = self.depth - i - 1
            
            # Unpool and concatenate
            res = xs[j]
            up  = torch.index_select(x, 0, cl[j])
            x   = torch.cat((res, up), dim=1)
            edge_index = ei[j]
            
            # Convolutions
            x = self.doGCNBlock(x, edge_index, self.up_convs[i])
        
        # Do the final convolution (if self.last_1x1 then remove graph edges)
        if self.last_1x1:
            edge_index = torch.empty((2,0), dtype=torch.long, device=self.get_device())
        x = self.final_conv(x, edge_index)
        
        # Apply a threshold to the output
        if self.threshold:
            x = F.threshold(x, self.threshold, self.threshold)
        
        # Return x
        return x
    
    # Need to fix the batch variable
    def buildBatch(self, x, edge_index):
        
        # If graphs have 3 nodes, batch is [0,0,0,1,1,1,2,2,2,...]
        batch = edge_index.new_zeros(x.size(0))
        n = self.Clusters[0].size(0)
        for i in range(int(x.size(0)/n)):
            batch[i*n:(i+1)*n] = i
        return batch
    
    # Function for applying GCN blocks
    def doGCNBlock(self, x, edge_index, block):
        
        # Loop through the convolutions and apply them
        for conv in block:
            x = conv(x, edge_index)
            x = self.activation(x)
        return x
    
    # Function for checking the cluster array
    def checkCluster(self, cluster, batch):
        
        # Cluster needs to be fixed if it's not the same size as batch
        if cluster.size(0) != batch.size(0):
            
            # If cluster is [0,1,2,1] change to [0,1,2,1, 3,4,5,4, ...]
            nn = cluster.size(0)
            bs = batch.max().item()+1
            nc = cluster.max().item()+1
            cluster = cluster.tile(bs)
            for i in range(bs):
                cluster[i*nn:(i+1)*nn] += (i*nc)
        
        return cluster
        
    # Function for returning the device the model is on
    def get_device(self):
        
        # Determine the device using the first key-value in the state_dict()
        dev = next(iter(self.state_dict().items()))[1].device
        return dev


# Make a class for 2D cnn models
class cnnModel2d(torch.nn.Module):

    def __init__(self, res, depth, channels_in, channels, convolutions, loss_function, activation, threshold, k_size, last_1x1, knn_up):
        super(cnnModel2d, self).__init__()
        self.type          = "cnn2d"
        self.res           = res
        self.depth         = depth
        self.channels_in   = channels_in
        self.channels      = channels
        self.convolutions  = convolutions
        self.loss_function = loss_function
        self.activation    = activation
        self.threshold     = threshold
        self.k_size        = k_size
        self.last_1x1      = last_1x1
        self.knn_up        = knn_up
        
        # Create initial convolutions
        chan_in  = channels_in
        chan_out = channels
        self.first_convs = self.buildCNNBlock(chan_in, chan_out)
        
        # Create the encoder side convolutions
        self.down_convs = torch.nn.ModuleList()
        for i in range(depth):
            chan_in  = chan_out
            chan_out = chan_out*2
            self.down_convs.append(self.buildCNNBlock(chan_in, chan_out))
        
        # Create the decoder side convolutions and transpose convolutions
        self.up_convs = torch.nn.ModuleList()
        self.tr_convs = torch.nn.ModuleList()
        for i in range(depth):
            chan_in  = chan_out
            chan_out = chan_out//2
            if self.knn_up:
                chan_in = chan_in + chan_out
            else:
                self.tr_convs.append(torch.nn.ConvTranspose2d(chan_in, chan_out, 2, stride=2))
            self.up_convs.append(self.buildCNNBlock(chan_in, chan_out))
        
        # Create the final convolution (maybe do a 1x1)
        chan_in  = channels
        chan_out = 1
        if self.last_1x1:
            self.final_conv = torch.nn.Conv2d(chan_in, chan_out, 1, stride=1, padding="same")
        else:
            self.final_conv = torch.nn.Conv2d(chan_in, chan_out, self.k_size, stride=1, padding="same")
        
        # Make sure to reset the parameters
        self.reset_parameters()
    
    # Use a function to add CNN blocks
    def buildCNNBlock(self, chan_in, chan_out):
        
        # Start a module list, build the block, and return it
        block = torch.nn.ModuleList()
        for i in range(self.convolutions):
            block.append(torch.nn.Conv2d(chan_in, chan_out, self.k_size, stride=1, padding="same"))
            chan_in = chan_out
        return block
    
    # Use a function to reset all the model parameters
    def reset_parameters(self):
        
        # Loop through all convolution module lists and reset
        for conv in self.first_convs:
            conv.reset_parameters()
        for list in self.down_convs:
            for conv in list:
                conv.reset_parameters()
        for conv in self.tr_convs:
            conv.reset_parameters()
        for list in self.up_convs:
            for conv in list:
                conv.reset_parameters()
        self.final_conv.reset_parameters()
    
    # Define the forward pass
    def forward(self, x, batch=None):
        
        # Define the pooling layer
        maxpool2x2 = torch.nn.MaxPool2d(2, stride=2)
            
        # Initialize list for storing skip connection info
        xs = []
        
        # Do the first convolutions
        x = self.doCNNBlock(x, self.first_convs)
        
        # Do the encoder side with pooling and convolutions
        for i in range(self.depth):
            
            # Pooling
            xs += [x]
            x   = maxpool2x2(x)
            
            # Convolutions
            x = self.doCNNBlock(x, self.down_convs[i])
        
        # Do the decoder side with upsampling and convolutions
        for i in range(self.depth):
            j = self.depth - i - 1
            
            # Unpool
            if self.knn_up:
                size_ = 2 * x.shape[1,2];
                x = F.interpolate(x, size_) # default mode="nearest"
            else:
                x = self.tr_convs[i](x)
            x = self.activation(x)
            
            # Concatenate
            res = xs[j]
            x   = torch.cat((x, res), dim=1)
            
            # Convolutions
            x = self.doCNNBlock(x, self.up_convs[i])
        
        # Do the final convolution
        x = self.final_conv(x)
        
        # Apply a threshold to the output
        if self.threshold:
            x = F.threshold(x, self.threshold, self.threshold)
        
        # Return x
        return x
    
    # Function for applying CNN blocks
    def doCNNBlock(self, x, block):
        
        # Loop through the convolutions and apply them
        for conv in block:
            x = conv(x)
            x = self.activation(x)
        return x
    
    # Function for returning the device the model is on
    def get_device(self):
        
        # Determine the device using the first key-value in the state_dict()
        dev = next(iter(self.state_dict().items()))[1].device
        return dev


# Make a class for 3D cnn models
class cnnModel3d(torch.nn.Module):

    def __init__(self, res, depth, channels_in, channels, convolutions, loss_function, activation, threshold, k_size, last_1x1, knn_up):
        super(cnnModel3d, self).__init__()
        self.type          = "cnn3d"
        self.res           = res
        self.depth         = depth
        self.channels_in   = channels_in
        self.channels      = channels
        self.convolutions  = convolutions
        self.loss_function = loss_function
        self.activation    = activation
        self.threshold     = threshold
        self.k_size        = k_size
        self.last_1x1      = last_1x1
        self.knn_up        = knn_up
        
        # Create initial convolutions
        chan_in  = channels_in
        chan_out = channels
        self.first_convs = self.buildCNNBlock(chan_in, chan_out)
        
        # Create the encoder side convolutions
        self.down_convs = torch.nn.ModuleList()
        for i in range(depth):
            chan_in  = chan_out
            chan_out = chan_out*2
            self.down_convs.append(self.buildCNNBlock(chan_in, chan_out))
        
        # Create the decoder side convolutions and transpose convolutions
        self.up_convs = torch.nn.ModuleList()
        self.tr_convs = torch.nn.ModuleList()
        for i in range(depth):
            chan_in  = chan_out
            chan_out = chan_out//2
            self.tr_convs.append(torch.nn.ConvTranspose3d(chan_in, chan_out, 2, stride=2))
            self.up_convs.append(self.buildCNNBlock(chan_in, chan_out))
        
        # Create the final convolutions
        chan_in  = channels
        chan_out = 1
        if self.last_1x1:
            self.final_conv = torch.nn.Conv3d(chan_in, chan_out, 1, stride=1, padding="same")
        else:
            self.final_conv = torch.nn.Conv3d(chan_in, chan_out, self.k_size, stride=1, padding="same")
        
        # Make sure to reset the parameters
        self.reset_parameters()
    
    # Use a function to add CNN blocks
    def buildCNNBlock(self, chan_in, chan_out):
        
        # Start a module list, build the block, and return it
        block = torch.nn.ModuleList()
        for i in range(self.convolutions):
            block.append(torch.nn.Conv3d(chan_in, chan_out, self.k_size, stride=1, padding="same"))
            chan_in = chan_out
        return block
    
    # Use a function to reset all the model parameters
    def reset_parameters(self):
        
        # Loop through all convolution module lists and reset
        for conv in self.first_convs:
            conv.reset_parameters()
        for list in self.down_convs:
            for conv in list:
                conv.reset_parameters()
        for conv in self.tr_convs:
            conv.reset_parameters()
        for list in self.up_convs:
            for conv in list:
                conv.reset_parameters()
        self.final_conv.reset_parameters()
    
    # Define the forward pass
    def forward(self, x, batch=None):
        
        # Define the pooling layer
        maxpool2x2x2 = torch.nn.MaxPool3d(2, stride=2)
        
        # Initialize list for storing skip connection info
        xs = []
        
        # Do the first convolutions
        x = self.doCNNBlock(x, self.first_convs)
        
        # Do the encoder side with pooling and convolutions
        for i in range(self.depth):
            
            # Pooling
            xs += [x]
            x   = maxpool2x2x2(x)
            
            # Convolutions
            x = self.doCNNBlock(x, self.down_convs[i])
        
        # Do the decoder side with upsampling and convolutions
        for i in range(self.depth):
            j = self.depth - i - 1
            
            # Unpool
            x = self.tr_convs[i](x)
            x = self.activation(x)
            
            # Concatenate
            res = xs[j]
            x   = torch.cat((x, res), dim=1)
            
            # Convolutions
            x = self.doCNNBlock(x, self.up_convs[i])
        
        # Do the final convolution
        x = self.final_conv(x)
        
        # Apply a threshold to the output
        if self.threshold:
            x = F.threshold(x, self.threshold, self.threshold)
        
        # Return x
        return x
        
    # Function for applying CNN blocks
    def doCNNBlock(self, x, block):
        
        # Loop through the convolutions and apply them
        for conv in block:
            x = conv(x)
            x = self.activation(x)
        return x
    
    # Function for returning the device the model is on
    def get_device(self):
        
        # Determine the device using the first key-value in the state_dict()
        dev = next(iter(self.state_dict().items()))[1].device
        return dev


# Function for displaying a summary of a model
def parameterSummary(model):
    
    # Print some info about the model
    printLine()
    print("        Device:", model.get_device())
    print("          Type:", model.type)
    print("         Depth:", model.depth)
    print("   Channels In:", model.channels_in)
    print("      Channels:", model.channels)
    print("  Convolutions:", model.convolutions)
    print("        ResNet:", model.res)
    print(" Loss Function:", model.loss_function)
    print(" Last Conv 1x1:", model.last_1x1)
    
    # Keep track of the total number of parameters
    total_trainable_parameters = 0
    
    # Build a nice table to print
    table = []
    for param in model.state_dict():
        sh = list(model.state_dict()[param].size())
        row = [param, sh, np.prod(sh)]
        table.append(row)
        total_trainable_parameters += row[2]
    
    # Print the table
    printLine()
    print(tabulate(table, headers=["Tensor Name", "Shape", "Size"], tablefmt="plain"))
    print("Total Trainable Parameters: {}".format(total_trainable_parameters))

