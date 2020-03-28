import torch


def save_checkpoint(model, train_data, save_directory):
    
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'input_size': 25088,
              'output_size': 1000,
              'hidden_layers': [4096,4096,2048],
              'state_dict': model.state_dict(),
              'optimizer': model.optimizer,
              'class_to_idx': model.class_to_idx,
              }

    torch.save(checkpoint, save_directory)
    print("Saved to",save_directory)


   