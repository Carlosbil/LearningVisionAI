import torch,os
from torchvision import datasets, transforms
from torch.utils.data import random_split, ConcatDataset
from DogNet import DogClassifier
from torch.utils.data import DataLoader
import torch.nn as nn

# Image transformations
transform = transforms.Compose([
    transforms.Resize((64,64)),  # Resizes the image to 64x64
    transforms.ToTensor(),  # Converts the image into a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizes the tensors (mean and std deviation for 3 color channels)
])

def target_transform_cifar(target):
    # All no dogs images are label 0
    return torch.tensor(0, dtype=torch.float32)

def target_transform_dogs(target):
    # All dog images are label 1
    return torch.tensor(1, dtype=torch.float32)

# Loading Stanford Dogs dataset
dogs_dataset = datasets.ImageFolder('./dataset/dogs/images', transform=transform, target_transform=target_transform_dogs)
num_samples_dogs = min(len(dogs_dataset), 10000)
dogs_dataset = torch.utils.data.Subset(dogs_dataset, indices=range(num_samples_dogs))

# Loading CIFAR-100 dataset
data_dir = './dataset/no_dogs/images/'
# Download and load CIFAR-100 dataset
cifar_dataset = datasets.CIFAR100(root=data_dir, download=True, transform=transform, target_transform=target_transform_cifar)
num_samples_cifar = min(len(cifar_dataset), 10000)
cifar_dataset = torch.utils.data.Subset(cifar_dataset, indices=range(num_samples_cifar))

# Concatenate both datasets
dataset = torch.utils.data.ConcatDataset([dogs_dataset, cifar_dataset])

# Split the data into training and testing sets
train_size = int(0.8 * len(dataset))  # 80% of data for training
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Path to save the model
model_path = 'dog_classifier.pth'
if os.path.exists(model_path):
    print("El modelo existe.")
    model = DogClassifier()
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        model = model.to('cuda')
else:
    print("El modelo no existe.")
    model_path = './dog_classifier.pth'
    model = DogClassifier()
    if torch.cuda.is_available():
        model = model.to('cuda')

criterion = nn.BCELoss()  # Binary Cross-Entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Define a loader for the training data
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=50, shuffle=True)

# Define a number of training epochs
epochs = 10

#actually is my best
best_loss =  0.5
# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    # Loop over each batch from the training set
    for i, (inputs, labels) in enumerate(train_loader):
        # Move tensors to GPU if CUDA is available
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
        
        # Clear the gradients of all optimized variables
        optimizer.zero_grad()
        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(inputs)
        # Calculate the batch loss
        loss = criterion(outputs, labels.unsqueeze(1))
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # Perform a single optimization step (parameter update)
        optimizer.step()
        
        # Update training loss
        running_loss += loss.item()
        # Print loss every 200 batches
        if i % 100 == 0:
            average_loss = running_loss / 100
            print('[Epoch %d, Batch %5d] Loss: %.7f' % (epoch + 1, i, average_loss))
            # Check if this is the best model so far
            if average_loss < best_loss:
                print('New best model found and saved!')
                best_loss = average_loss
                # Save the model
                torch.save(model.state_dict(), model_path)    
            running_loss = 0.0

# Initialize counters
correct = 0
total = 0
# No need to track gradients for testing
with torch.no_grad():
    model.eval()
    # Loop over each batch from the test set
    for images, labels in test_loader:
        # Move tensors to GPU if CUDA is available
        if torch.cuda.is_available():
            images = images.to('cuda')
            labels = labels.to('cuda')
        
        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = model(images)
        # Convert outputs to binary predictions (0 or 1)
        predicted = outputs.round()
        # Update total and correct counters
        total += labels.size(0)
        correct += (predicted == labels.unsqueeze(1)).sum().item()

# Calculate and print the accuracy
print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
