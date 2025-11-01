import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model_cnn_def import CNN

#Transformations - convert to tensor & normalize
transform = transforms.Compose([
    transforms.RandomRotation(10), # rotate by +/- 10 degrees
    transforms.RandomAffine(0, translate=(0.1, 0.1)), # translate by +/- 10% in both directions
    transforms.RandomResizedCrop(28, scale=(0.9, 1.1)), # random crop and resize to 28x28
    transforms.RandomPerspective(distortion_scale=0.3, p=0.5), # perspective transform
    transforms.ToTensor(),                # convert image to tensor
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)), # random erasing a patch from image for model to learn global structure of digits
    transforms.Normalize((0.1307,), (0.3081,))  # mean and std of MNIST
])

#Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset  = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

#DataLoader - for batching
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize CNN model
model = CNN()
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()     # suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10  

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    if epoch % max(1, (num_epochs//10)) == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# Evaluate on test data
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
    

print("Training complete")
# Save the trained model
torch.save(model.state_dict(), "CNN_PyTorch/mnist_cnn.pth")
print("Model saved as mnist_cnn.pth")








