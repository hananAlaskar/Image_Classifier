
from load_data import load_data
from set_parameters_for_trining import set_parameters_for_trining
from train_model import train
from save_checkpoint import save_checkpoint


def main():

    data_dir = "flowers_10"
    trainloader, validloader, train_data = load_data(data_dir)

    lr = 0.001   
    model,criterion,optimizer = set_parameters_for_trining(lr)

    epochs = 6
    model = train(model, trainloader, validloader, criterion, optimizer, epochs)
    
    save_directory = "checkpoint.pth"
    save_checkpoint(model, train_data, save_directory)
    
# Call to main function to run the program
if __name__ == "__main__":
    main()






