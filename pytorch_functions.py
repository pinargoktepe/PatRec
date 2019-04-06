import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch


# Definition of the model
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes):
        super(SimpleModel, self).__init__()
        self.main = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=False),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, n_classes, bias=False))

    def forward(self, input):
        out = input.view(input.size(0), -1)
        out = self.main(out)
        return out


# Function that plot train and validation losses, errors and accuracies
def plot(n_epochs, train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure()
    plt.plot(np.arange(n_epochs), train_losses)
    plt.plot(np.arange(n_epochs), val_losses)
    plt.legend(['train_loss', 'val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.title('Train/val loss')
    plt.show();

    train_err = 100 * np.ones((len(train_accuracies))) - train_accuracies
    val_err = 100 * np.ones((len(val_accuracies))) - val_accuracies
    plt.figure()
    plt.plot(np.arange(n_epochs), train_err)
    plt.plot(np.arange(n_epochs), val_err)
    plt.legend(['train_error', 'val_error'])
    plt.xlabel('Epoch')
    plt.ylabel('Error [%]')
    plt.title('Train/val error')
    plt.show();

    plt.figure()
    plt.plot(np.arange(n_epochs), train_accuracies)
    plt.plot(np.arange(n_epochs), val_accuracies)
    plt.legend(['train_acc', 'val_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.title('Train/val accuracy')
    plt.show();


# Train network
def train(model, train_dataloader, val_dataloader, optimizer, n_epochs, loss_function):
    # We will monitor loss functions as the training progresses
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(n_epochs):
        ## Training phase
        model.train()
        correct_train_predictions = 0 # We will measure accuracy
        # Iterate mini batches over training dataset
        losses = []
        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images) # run prediction; output <- vector with probabilities of each class
            # set gradients to zero
            optimizer.zero_grad()
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()

            # Metrics
            losses.append(loss.item())
            predicted_labels = output.argmax(dim=1)
            n_correct = (predicted_labels == labels).sum().item()
            correct_train_predictions += n_correct
        train_losses.append(np.mean(np.array(losses)))
        train_accuracies.append(100.0*correct_train_predictions/len(train_dataloader.dataset))

        # Evaluation phase
        model.eval()
        correct_val_predictions = 0 
        # Iterate mini batches over validation set
        losses = []
        with torch.no_grad():
            for images, labels in val_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                loss = loss_function(output, labels)

                losses.append(loss.item())
                predicted_labels = output.argmax(dim=1)
                n_correct = (predicted_labels == labels).sum().item()
                correct_val_predictions += n_correct
        val_losses.append(np.mean(np.array(losses)))
        val_accuracies.append(100.0*correct_val_predictions/len(val_dataloader.dataset))

        print('Epoch {}/{}: train_loss: {:.4f}, train_accuracy: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.4f}'.format(epoch+1, n_epochs,
                                                                                                      train_losses[-1],
                                                                                                      train_accuracies[-1],
                                                                                                      val_losses[-1],
                                                                                                      val_accuracies[-1]))
    return train_losses, val_losses, train_accuracies, val_accuracies


# Test model with testdataset
def test(dataloader, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        correct_predictions = 0
        for images, labels in dataloader:
            images = images.to(device)
            #labels = labels.to(device)
            output = model(images)
            predicted_labels = output.argmax(dim=1)
            n_correct = (predicted_labels == labels).sum().item()
            correct_predictions += n_correct
        accuracies = 100.0*correct_predictions/len(dataloader.dataset)
    return accuracies

