# test_pytorch.py
import torch
from torchvision import transforms
from PIL import Image
from model_def import DigitClassifier
import matplotlib.pyplot as plt

# Load trained model
model = DigitClassifier()
model.load_state_dict(torch.load("PyTorch_NN/mnist_model.pth",weights_only=True))
model.eval()

# Define same preprocessing as training
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Path to custom image
image_path = "/Users/kartikayasrivastava/Desktop/Digit_24oct/digits/four.jpg"  # <-- Change this to your image path
image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # Add batch dimension

plt.imshow(image[0].squeeze().numpy(), cmap="gray")
plt.title("Image seen by model (after preprocessing)")
plt.axis("off")
plt.show()

# Run inference
with torch.no_grad():
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probabilities, 1)

print(f"Predicted Digit: {predicted.item()}")
print(f"Confidence: {confidence.item()*100:.2f}%")
