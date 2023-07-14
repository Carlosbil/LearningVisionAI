import torch
import torch.nn as nn
import torch.nn.functional as F

class DogClassifier(nn.Module):
    def __init__(self):
        super(DogClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # Nueva capa convolucional
        self.dropout = nn.Dropout(0.25)  # Capa de dropout
        self.fc1 = nn.Linear(64*8*8, 512)  # Ajuste en la cantidad de características de entrada
        self.fc2 = nn.Linear(512, 1)
        self.batch_norm1 = nn.BatchNorm2d(16)  # Capa de normalización de lotes
        self.batch_norm2 = nn.BatchNorm2d(32)  # Capa de normalización de lotes
        self.batch_norm3 = nn.BatchNorm2d(64)  # Capa de normalización de lotes
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.batch_norm1(self.conv1(x))), 2)  
        x = F.max_pool2d(F.relu(self.batch_norm2(self.conv2(x))), 2)
        x = F.max_pool2d(F.relu(self.batch_norm3(self.conv3(x))), 2)  # Nueva capa convolucional
        x = x.view(-1, 64*8*8)  # Ajuste en la cantidad de características de entrada
        x = self.dropout(x)  # Capa de dropout
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
