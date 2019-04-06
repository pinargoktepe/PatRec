import numpy as np
import torch
from pytorch_functions import plot, train, test, SimpleModel
import torch.nn as nn
import torchvision
import torchvision.transforms as TF
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Define batch_size
batch_size = 64

# Load datasets
train_data = np.loadtxt(open('mnist_train.csv','r'), delimiter=',', skiprows=0)  # Load dataset
test_data = np.loadtxt(open('mnist_test.csv','r'), delimiter=',', skiprows=0)  # Load dataset

# Separate images and labels
images = train_data[:,1:]
labels = train_data[:,0]
test_images = test_data[:,1:]
test_labels = test_data[:,0]

# Reshape train images
data = np.zeros((images.shape[0],28,28))
for i in np.arange(0,images.shape[0]):
    data[i] = np.reshape(images[i,:], (28, 28))
images = np.double(data)

# Reshape test images
data = np.zeros((test_images.shape[0],28,28))
for i in np.arange(0,test_images.shape[0]):
    data[i:i+1] = np.reshape(test_images[i,:], (28, 28))
test_images = np.double(data)

# Split dataseet into train and validation
proctr = 90 # How much percentages of whole data is for training
nu = int(proctr*images.shape[0]/100)

train_images = images[:nu,:]
val_images = images[nu:,:]
train_labels = labels[:nu]
val_labels = labels[nu:]


#change to tensors
x = torch.from_numpy(train_images)
y = torch.from_numpy(train_labels)
#create dataset and use data loader
train_dataset = TensorDataset(x.float(), y.long())

#change to tensors
x = torch.from_numpy(val_images)
y = torch.from_numpy(val_labels)
#create dataset and use data loader
val_dataset = TensorDataset(x.float(), y.long())

#change to tensors
x = torch.from_numpy(test_images)
y = torch.from_numpy(test_labels)
#create dataset and use data loader
test_dataset = TensorDataset(x.float(), y.long())

# Make dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=64)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=64)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=64)

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: ' + ('gpu' if torch.cuda.is_available() else 'cpu'))

# Define hyperparameters
learning_rate = 0.001  	# Learning rate
n_epochs = 25			# Number of epochs
n_in = 3				# Number of random initializations

# Start Training
val_old = 0
for i in np.arange(0,n_in):
    print("Initalization "+str(i+1)+"/"+str(n_in))
	
	# Generate new model with random weights
    model = SimpleModel(input_dim=28 * 28 * 1, hidden_dim=20, n_classes=10)  # dimensionality of x hidden dimension 20
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
	# Train it
    train_losses, val_losses, train_accuracies, val_accuracies = train(model, train_dataloader, val_dataloader, optimizer, n_epochs, loss_function)
	# Save best model
    if val_accuracies[-1]>val_old:
        val_old = val_accuracies[-1]
        best_i = i
        best_mod = model
        f_train_losses, f_val_losses, f_train_accuracies, f_val_accuracies = train_losses, val_losses, train_accuracies, val_accuracies


print("Best model at initalization "+str(i+1))
plot(n_epochs, f_train_losses, f_val_losses, f_train_accuracies, f_val_accuracies) 	# plot graphs

# Get test accuracy and print it
test_acc = test(test_dataloader, best_mod)
print('Test accuracy: ' + str(test_acc))
