import argparse


def get_train_input_args():
   
    
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type = str, 
                        help = 'path to the folder of data') 
    
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', 
                        help = 'path to the folder of data') 
    
    parser.add_argument('--learning_rate', type=float, default = 0.0001, 
                        help = 'Learning Rate')
    
    parser.add_argument('--epochs', type=int, default = 10, 
                        help = 'number of epochs')

    print("Argument Info")
    print("data_dir : {}".format(parser.parse_args().data_dir))
    print("save_dir : {}".format(parser.parse_args().save_dir))
    print("learning_rate : {}".format(parser.parse_args().learning_rate))
    print("epochs : {}".format(parser.parse_args().epochs))

    
    return parser.parse_args()
