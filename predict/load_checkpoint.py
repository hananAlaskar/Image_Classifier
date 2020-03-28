import torch
import torchvision.models as models
from classifier import Classifier



def load_checkpoint(filepath): 
    
    checkpoint = torch.load(filepath)
    load_model = models.vgg16(pretrained=True)
    load_model.classifier = Classifier(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'])
    
    load_model.optimizer = checkpoint['optimizer']
    load_model.class_to_idx = checkpoint['class_to_idx']
                
    load_model.load_state_dict(checkpoint['state_dict'])
    
    return load_model
   