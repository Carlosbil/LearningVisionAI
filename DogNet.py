import torch
import torch.nn as nn
import torch.nn.functional as F

class DogClassifier(nn.Module):
    def __init__(self):
        super(DogClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # Convolutional layer (3 input channels (RGB), 16 output channels, 3x3 kernel)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # Convolutional layer (16 input channels, 32 output channels, 3x3 kernel)
        self.fc1 = nn.Linear(32*16*16, 512)  # Fully connected layer (32*16*16 input features, 512 output features)
        self.fc2 = nn.Linear(512, 1)  # Fully connected layer (512 input features, 1 output feature)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # First conv layer followed by ReLU activation function and MaxPooling
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # Second conv layer followed by ReLU activation function and MaxPooling
        x = x.view(-1, 32*16*16)  # Flattens the data for the fully connected layers
        x = F.relu(self.fc1(x))  # First fc layer followed by ReLU activation function
        x = torch.sigmoid(self.fc2(x))  # Second fc layer followed by sigmoid activation function (for binary classification)
        return x
