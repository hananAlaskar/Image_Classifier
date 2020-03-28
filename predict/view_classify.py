
from get_classes import get_classes,get_class_index


def view_classify(probs, classes_indexs, model, file_name):
  
   classes_keys = get_class_index(classes_indexs, model)
   if len(classes_keys) != 0:
    classes = get_classes(classes_keys, file_name)
    classes_probs = list(zip(classes,probs))
    for item in classes_probs:   
        print(item)
    
