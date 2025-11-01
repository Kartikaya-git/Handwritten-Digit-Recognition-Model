import torch
from torchvision import datasets, transforms
from model_cnn_def import CNN
from PIL import Image

# 1. Load your trained model
model = CNN()  
model.load_state_dict(torch.load("mnist_cnn.pth",weights_only=True, map_location=torch.device('cpu')))
model.eval()

# 2. Load your image
img_path = "/Users/kartikayasrivastava/Desktop/Digit_24oct/digits/two.jpg"   # <-- change this to your file name
image = Image.open(img_path) 

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

image = transform(image)
image = image.unsqueeze(0)  # add batch dimension: [1,1,28,28]

# 4. Predict
with torch.no_grad():
    output = model(image)
    predicted = torch.argmax(output, dim=1).item()

print(f"Predicted Digit: {predicted}")