import torch
from torch import nn
import torch.nn.functional as F
    
''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
'''
    
class Classifier(nn.Module):
    def __init__(self, input_size = 25088, output_size = 1000, hidden_layers = [4096,4096,2048], drop_p=0.5):
        super().__init__()
        
        self.input = nn.Linear(input_size, hidden_layers[0])
       
          # Add a variable number of more hidden layers
        self.hidden_layers = [] 
        for index in range(len(hidden_layers)-1):
            self.hidden_layers.append(nn.Linear(hidden_layers[index],hidden_layers[index+1]))
            self.add_module("hidden_layer_"+str(index), self.hidden_layers[-1])

        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p = drop_p)

    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.input(x)))

        for each in self.hidden_layers:
            x = self.dropout(F.relu(each(x)))
            
        x = F.log_softmax(self.output(x), dim=1)
   
        return x
