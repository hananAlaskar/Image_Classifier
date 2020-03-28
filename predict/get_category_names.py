import json

def get_category_names(file_name):
    with open(file_name, 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name