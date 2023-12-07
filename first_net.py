# %% importing packages
import torch
import torch.nn as nn 
import numpy as np

# %% defining the network
class JNet(nn.Module):
    def __init__(self, actv, input_feature_num, hidden_unit_nums, output_feature_num):
        # actv: the actvation function in the neurons (how will each neuron transform the signal)
        # input feature num: this has to have the same shape fo the input for the netwrok.
        # Ex.: if it is a RGB image with three channels, it needs to be a 32 x 32 x 3 input feature num.
        # if it is matrices with 30 colunms and 3 rows, the input feature num will be (3, 30)

        # hidden unit nums:
        # in this an array that specifies the number of neurons per layer and the number of layers need to be specified.
        # in this way the lenghth of the array will be the amount of hidden layers and the number in each of the elements of the array
        # will specify how many neurons per neuron there are.

        # output feature num: specify the format of the output, If it is categories, yes or no, anything.

        # now, we have to initialize the nn module as well, because we ned the initializations from this module
        super(JNet, self).__init__()
        self.input_feature_num = input_feature_num # Save the input size for reshaping later
        self.mlp = nn.Sequential() # This makes sure that the network will process one layer after the another and we will not 
                                    # have to make this process manually
        
        in_num = input_feature_num
        for in in range(len(hidden_unit_nums)):
            # this will loop throught all the hidden layers specified in the object call.
            # if we know the number of layers we will have in our network, 
            # we can specify in the sequential function.
            # Althought, the purpose of this code is to create a general MLP 
            # and to be the most generalizable possible

            out_num = hidden_unit_nums[i] 
            # this will read how many neurons are in the ith layer of the network
            linear = ...
            in_num = out_num 
            # because the input for the next layer is the output from the previous one, 
            # it is necessary to define the input_num as the output num

            self.mlp.add_module('Linear_%d'%i, layer) 
            # Append layer to the model with a name
            # this will basically just add, officially, the layer to the model that we are building

            actv_layer = eval('nn.%s'%actv) 
            # Assign activation function (eval allows us to instantiate object from string)
            # this will get the string that was passed to the function as activation function to the layer 
            # and will make it an object from the module nn. 
            # Like nn."the activation function specified in the call"

            self.mlp.add_module('Activation_%d'%i, actv_layer) 
            # Append activation to the model with a name
            # this is also basically oficcialy appendind the activation function of this layer to the model. 
            # Making it official

            out_layer = nn.Linear(in_num, output_feature_num) 
            # Create final layer. This is the output layer. 
            # It is created separatelly because it does not have an actvation function. 
            self.mlp.add_module('Output_Linear', out_layer) 
            # Append the final layer
            # add the output layer in order to the module.

    def forward(self, x):
        # now, make the forward pass of the model.
        # the forward pass means that the inputs will gio through the netwrok once 
        # and an input will be delivered
        """
        Simulate forward pass of MLP Network

        Args:
        x: torch.tensor
            Input data

        Returns:
        logits: Instance of MLP
            Forward pass of MLP
        """
        # Reshape inputs to (batch_size, input_feature_num)
        # Just in case the input vector is not 2D, like an image!
        x = x.view(-1, self.input_feature_num)

        ####################################################################
        # Fill in missing code below (...),
        # then remove or comment the line below to test your function
        # raise NotImplementedError("Run MLP model")
        ####################################################################

        logits = ... # Forward pass of MLP
        return logits
            


        