import torchvision.models as models
import torch
from torch import nn
from torch import optim

from classifier import Classifier

def set_parameters_for_trining(lr):
    
    model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    
    classifier =  Classifier() 
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    model.optimizer = optimizer.state_dict

    
    return model,criterion,optimizer
