# %%
# importing packages
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# %%

# this code was based on the MLP exercises of Neuromatch academy.
# the file "first network.py" contains the whole code.
# it will be modified from a general MPL to a MPL shapped to accept only the input size for CS categorization
# and input size will be two (0 or 1)
# creating arcjitecture of the network
# general first before i know the size of the three inputs that will contain

class csNet(nn.Module):
    def __init__(self, actv_fun, input_num, hidden_layer_defs, outputs_num):
        
        """
        actv_fun: which activation function will be used by layer or by network
        input_num: get the shape of the cs data. 
        It three values are considered: milivolts before the onset of the spike, 
        milivolts at the peak time, time of standardized minimum time.
        OBS: look for the measures inside the epics network
        """

        self.input_feature_num = input_num # input size if needs reshapping.
        self.mlp = nn.Sequential() 
        
        # initialize this module so I do not have to specify manually 
        # the sequence of layers.
        # maybe change the architecture or the network as well.

        out_num = hidden_layer_defs[i] # get the number of outputs considering the layer size
        layer = nn.Linear(in_num, out_num) # Use nn.Linear to define current layer
        in_num = out_num # input from next layer is output from previous one
        self.mlp.add_module('Linear_%d'%i, layer) # name the layer just created

