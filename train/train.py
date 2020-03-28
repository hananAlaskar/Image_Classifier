
from get_train_input_args import get_train_input_args
from load_data import load_data
from set_parameters_for_trining import set_parameters_for_trining
from train_model import train
from save_checkpoint import save_checkpoint


def main():

    in_arg = get_train_input_args()

    data_dir = in_arg.data_dir
    trainloader, validloader, train_data = load_data(data_dir)

    lr = in_arg.learning_rate
    model,criterion,optimizer = set_parameters_for_trining(lr)

    epochs = in_arg.epochs
    model = train(model, trainloader, validloader, criterion, optimizer, epochs)
    
    save_directory = in_arg.save_dir
    save_checkpoint(model, train_data, save_directory)
    
# Call to main function to run the program
if __name__ == "__main__":
    main()






