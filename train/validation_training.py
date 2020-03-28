import torch


def validation(model, validloader, criterion):
    
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
            for images, labels in validloader:

                log_ps = model.forward(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                op_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))     
                    
            return test_loss, accuracy
        