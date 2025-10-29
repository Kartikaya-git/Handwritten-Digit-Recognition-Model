import numpy as np
#from PIL import Image
from Numpy_version.train_digit_no_lib import forward_prop
import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28,28))
    _, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    img_array = np.array(img, dtype=np.float32)
    #img_array = 255 - img_array
    img_array = img_array /255.0
    img_array = img_array.reshape(784,1)
    return img_array


def preprocess_image_cv(image_path):
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Invert so digit becomes white, background black
    img = 255 - img

    # Threshold to get a binary image (digit = white, bg = black)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Find bounding box of the digit
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]

    # Resize digit keeping aspect ratio
    if w > h:
        new_w = 20
        new_h = int(h * (20 / w))
    else:
        new_h = 20
        new_w = int(w * (20 / h))
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad to 28x28
    top = (28 - new_h) // 2
    bottom = 28 - new_h - top
    left = (28 - new_w) // 2
    right = 28 - new_w - left
    digit_padded = cv2.copyMakeBorder(
        digit_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=0
    )

    # ✅ Final inversion so background is black and digit is white (MNIST-style)
    digit_final = 255 - digit_padded

    # Normalize to [0,1] and reshape
    digit_final = digit_final.astype(np.float32) / 255.0
    return digit_final.reshape(784, 1)



data = np.load("/Users/kartikayasrivastava/Desktop/Digit_24oct/mnist_weights1.npz")
w1 = data["w1"]
b1 = data["b1"]
w2 = data["w2"]
b2 = data["b2"]
x_input = preprocess_image_cv('/Users/kartikayasrivastava/Desktop/Digit_24oct/digits/one.jpg')
_, _, _, a2 = forward_prop(w1, b1, w2, b2, x_input)
predicted_digit = np.argmax(a2)
confidence = a2[predicted_digit, 0] * 100
print(f"Predicted digit: {predicted_digit} with {confidence:.2f}% confidence")

import matplotlib.pyplot as plt

plt.imshow(x_input.reshape(28,28), cmap='gray')
plt.show()