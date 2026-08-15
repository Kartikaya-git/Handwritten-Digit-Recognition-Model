"""Training-time augmentation.

Three groups:

1. Custom real-world damage (hand-written, PIL-level): random occlusions,
   partial erasures, noise injection.
2. Geometric jitter, ported from the MNIST CNN in ../CNN_PyTorch: rotation,
   translation, scale, perspective, random erasing. `fill=255` everywhere --
   these images are dark ink on white paper, so torchvision's default
   black fill would paste fake black borders into every rotated sample.
3. Photometric jitter (brightness/contrast/blur). The MNIST version had no
   equivalent and did not need one: MNIST is already clean, uniform,
   digitally-rendered. This dataset gets used against phone photos of
   paper, where exposure and focus vary a lot.
"""
import random

import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms

from model_cnn_def import IMG_SIZE

FILL = 255  # white paper


class RandomOcclusion:
    """Paints small random black/white blobs over the symbol to simulate
    ink smudges or objects partially covering a written symbol."""

    def __init__(self, p=0.3, max_patches=2, max_frac=0.18):
        self.p = p
        self.max_patches = max_patches
        self.max_frac = max_frac

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img = img.copy()
        w, h = img.size
        draw = ImageDraw.Draw(img)
        n = random.randint(1, self.max_patches)
        for _ in range(n):
            pw = random.uniform(0.08, self.max_frac) * w
            ph = random.uniform(0.08, self.max_frac) * h
            x0 = random.uniform(0, w - pw)
            y0 = random.uniform(0, h - ph)
            fill = random.choice([0, 255])
            draw.rectangle([x0, y0, x0 + pw, y0 + ph], fill=fill)
        return img


class RandomPartialErasure:
    """Erases a chunk near an edge of the symbol to simulate a partially
    erased or cut-off stroke."""

    def __init__(self, p=0.25, max_frac=0.25):
        self.p = p
        self.max_frac = max_frac

    def __call__(self, img):
        if random.random() > self.p:
            return img
        img = img.copy()
        w, h = img.size
        draw = ImageDraw.Draw(img)
        side = random.choice(["left", "right", "top", "bottom"])
        frac = random.uniform(0.12, self.max_frac)
        if side == "left":
            box = [0, 0, w * frac, h]
        elif side == "right":
            box = [w * (1 - frac), 0, w, h]
        elif side == "top":
            box = [0, 0, w, h * frac]
        else:
            box = [0, h * (1 - frac), w, h]
        draw.rectangle(box, fill=255)
        return img


class RandomNoiseInjection:
    """Adds Gaussian + salt-and-pepper noise to a [0,1] float tensor."""

    def __init__(self, p=0.4, gaussian_std=0.06, salt_pepper_frac=0.01):
        self.p = p
        self.gaussian_std = gaussian_std
        self.salt_pepper_frac = salt_pepper_frac

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        noise = np.random.normal(0, self.gaussian_std, tensor.shape).astype("float32")
        out = tensor.numpy() + noise
        mask = np.random.random(out.shape) < self.salt_pepper_frac
        salt = np.random.random(out.shape) < 0.5
        out[mask & salt] = 1.0
        out[mask & ~salt] = 0.0
        import torch

        return torch.from_numpy(out.clip(0.0, 1.0).astype("float32"))


def build_train_augment():
    """PIL-level augmentation applied AFTER the shared base_transform."""
    return transforms.Compose([
        # group 1 -- custom real-world damage
        RandomOcclusion(p=0.3),
        RandomPartialErasure(p=0.25),
        # group 2 -- geometric, ported from ../CNN_PyTorch/train_cnn.py
        transforms.RandomRotation(10, fill=FILL),
        transforms.RandomAffine(0, translate=(0.1, 0.1), fill=FILL),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5, fill=FILL),
        transforms.RandomResizedCrop(
            IMG_SIZE, scale=(0.85, 1.1), ratio=(0.9, 1.1), antialias=True
        ),
        # group 3 -- photometric, for phone-photo robustness
        transforms.ColorJitter(brightness=0.35, contrast=0.35),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
    ])
