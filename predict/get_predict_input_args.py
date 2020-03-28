import argparse


def get_predict_input_args():
   
    
    parser = argparse.ArgumentParser()
    
        
    parser.add_argument('image_path', type = str, 
                        help = 'image path')
    
    parser.add_argument('checkpoint', type = str,
                        help = 'path to the folder of data') 
    
    parser.add_argument('--top_k', type=int, default = 5, 
                        help = 'top k')
    
    parser.add_argument('--category_names', default = 'cat_to_name_10.json', 
                        help = 'category names')


    print("Argument Info")
    print("image_path : {}".format(parser.parse_args().image_path))
    print("checkpoint : {}".format(parser.parse_args().checkpoint))
    print("top k : {}".format(parser.parse_args().top_k))
    print("category names : {}".format(parser.parse_args().category_names))

    
    return parser.parse_args()
