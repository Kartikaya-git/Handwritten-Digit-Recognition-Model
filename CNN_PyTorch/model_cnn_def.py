import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 1 input channel (grayscale), 32 output filters, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64*5*5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Layer 1: Conv + ReLU + Pool
        x = F.relu(self.conv1(x))     # (N, 32, 26, 26)
        x = F.max_pool2d(x, 2)        # (N, 32, 13, 13)
        
        # Layer 2: Conv + ReLU + Pool
        x = F.relu(self.conv2(x))     # (N, 64, 11, 11)
        x = F.max_pool2d(x, 2)        # (N, 64, 5, 5)
        
        # Flatten
        x = x.view(-1, 64*5*5)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

