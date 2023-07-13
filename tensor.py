import torch,os
from torchvision import datasets, transforms
from torch.utils.data import random_split, ConcatDataset
from DogNet import DogClassifier
from torch.utils.data import DataLoader
import torch.nn as nn
# Creamos un tensor de ceros de tamaño 3x3
x = torch.zeros(3, 3)
print(x)
# Verifica si CUDA (la API de GPU de NVIDIA) está disponible, luego mueve el tensor a GPU si es posible

if torch.cuda.is_available():
    x = x.to('cuda')
    print("using cuda")
# Sumamos 1 a todos los elementos del tensor
x += 1
print(x)

# Image transformations
transform = transforms.Compose([
    transforms.Resize((64,64)),  # Resizes the image to 64x64
    transforms.ToTensor(),  # Converts the image into a tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizes the tensors (mean and std deviation for 3 color channels)
])

def target_transform_cifar(target):
    # Asignar todas las etiquetas como 2
    return 1
def target_transform_dogs(target):
    # Asignar todas las etiquetas como 2
    return 0

# Loading Stanford Dogs dataset
dogs_dataset = datasets.ImageFolder('./dataset/dogs/images', transform=transform, target_transform=target_transform_dogs)

# Loading CIFAR-100 dataset
data_dir = './dataset/no_dogs/images/'
# Download and load CIFAR-100 dataset
cifar_dataset = datasets.CIFAR100(root=data_dir, download=True, transform=transform, target_transform=target_transform_cifar)

# Concatenate both datasets
dataset = torch.utils.data.ConcatDataset([dogs_dataset, cifar_dataset])

# Split the data into training and testing sets
train_size = int(0.8 * len(dataset))  # 80% of data for training
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

class_names = list(set([class_name for _, class_name in train_dataset]))
print(class_names)
#after obtain the datasets, create the model
# Path to save the model
model_path = 'DogNet.pth'
if os.path.exists(model_path):
    print("El modelo existe.")
    model = DogClassifier()
    model.load_state_dict(torch.load(model_path))
else:
    print("El modelo no existe.")
    model_path = './dog_classifier.pth'
    model = DogClassifier()
    if torch.cuda.is_available():
        model = model.to('cuda')

criterion = nn.BCELoss()  # Binary Cross-Entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer

# Define a loader for the training data
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define a number of training epochs
epochs = 10




# Training loop
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
        
        # Reset the gradients
        optimizer.zero_grad()
        # Forward pass (predict the labels)
        outputs = model(inputs)
        # Calculate loss
        loss = criterion(outputs, labels.float().unsqueeze(1))
        # Backward pass (compute gradients)
        loss.backward()
        # Update the parameters
        optimizer.step()
        # Print loss statistics
        running_loss += loss.item()
        if i % 200 == 199:    # Print every 200 mini-batches
            print('[Epoch %d, Batch %5d] Loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

    # Save the model after each epoch
    torch.save(model.state_dict(), model_path)