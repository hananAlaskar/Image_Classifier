from get_category_names import get_category_names


def get_classes(keys, file_name):
    category_names = get_category_names(file_name)
    return[category_names['{}'.format(x)] for x in keys]


def get_key(val, model): 
    for key, value in model.class_to_idx.items(): 
         if val == value: 
                return key 
    return 0

def get_class_index(indexs, model):
    keys = []
    for index in indexs:
        key = get_key(index, model)
        if key != 0:
            keys.append(key)
        
    return keys