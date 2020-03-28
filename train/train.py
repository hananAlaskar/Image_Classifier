from load_data import load_data


def main():

    data_dir = "flowers_10"
    trainloader, validloader, train_data = load_data(data_dir)
    

# Call to main function to run the program
if __name__ == "__main__":
    main()






