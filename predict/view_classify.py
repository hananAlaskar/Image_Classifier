import matplotlib.pyplot as plt
import matplotlib
import numpy as np


from get_classes import get_classes,get_class_index
from process_image import process_image
from undo_preprocessing import undo_preprocessing


def view_classify(probs, classes_indexs, model, file_name, image_path):

    matplotlib.get_backend()
  
    classes_keys = get_class_index(classes_indexs, model)
    if len(classes_keys) != 0:
        classes = get_classes(classes_keys, file_name)
        classes_probs = list(zip(classes,probs))
        print("Predict Result")
        for item in classes_probs:   
            print(item)

        classes_size = len(classes_keys)
        fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
        ax1.set_title("Predict Result")
        ax1.barh(np.arange(classes_size),probs)
        ax1.set_aspect(np.mean(probs))
        ax1.set_yticks(np.arange(classes_size))
        ax1.set_yticklabels(classes, size='small');
        ax1.set_xlim(0,np.max(probs))
        ax2.set_title("Predicted Image")
        image = undo_preprocessing(process_image(image_path))
        ax2.imshow(image)
        ax2.axis('off')

        plt.tight_layout()
        plt.show(block=True)
    
