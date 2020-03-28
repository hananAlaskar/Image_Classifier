import torch

from process_image import process_image


def predict(image_path, model, top_k):
    image_array = process_image(image_path)
    image_torch = torch.from_numpy(image_array).unsqueeze(0)
        
    with torch.no_grad():
        logps = model.forward(image_torch)
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(top_k, dim=1)
    
    return top_p, top_class
    
