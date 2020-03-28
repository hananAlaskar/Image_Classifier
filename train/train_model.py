import torch
import time

from validation_training import validation

def train(model, trainloader, validloader, criterion, optimizer, epochs, print_every=3):
    
    print("Start Training")  

    start = time.time() 

    steps = 0
    running_loss = 0
    train_losses = []

    for epoch in range(epochs):
        for images, labels in trainloader:

            steps += 1
                   
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
                    
            if steps % print_every == 0:
                test_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                "Test Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                "Test Accuracy: {:.3f}".format(accuracy/len(validloader)))
                
                running_loss = 0

                model.train()

              

    model.epochs = epochs
    print("End Training")     
    totalTime = time.time() - start 
    print("The total training time: {:.3f} minuates".format(totalTime/60))
    
    return model
    
    
    

