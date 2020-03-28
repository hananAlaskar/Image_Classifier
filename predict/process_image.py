import torch
import numpy as np
import torchvision

from torchvision import transforms
from PIL import Image

def process_image(image):
    

    image_transforms = torchvision.transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    img_pil = Image.open(image)
    img_tensor = image_transforms(img_pil)
    img_array = img_tensor.numpy()
    img_array = img_array.transpose((0, 1, 2))
    
    return img_array
     