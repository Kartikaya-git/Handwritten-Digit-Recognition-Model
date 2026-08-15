"""Shared preprocessing. Anything here runs at BOTH training and inference
time -- keeping them in one place is deliberate, since a train/inference
preprocessing mismatch is invisible in benchmark numbers but wrecks real
predictions.
"""
import numpy as np
from PIL import Image, ImageOps

from model_cnn_def import IMG_SIZE

# Measured over all 7,750 images of this dataset after the deterministic
# base transform below. The MNIST-era code hardcoded MNIST's
# (0.1307, 0.3081) -- inverted and wrong for this data, which is mostly
# white paper (high mean) rather than MNIST's mostly-black background.
NORM_MEAN = 0.9435
NORM_STD = 0.2053


def to_square(img, fill=255):
    w, h = img.size
    side = max(w, h)
    return ImageOps.pad(img, (side, side), color=fill, centering=(0.5, 0.5))


def stretch_contrast(pil_img, cutoff=2):
    """Rescale so the symbol's own dark/light range spans the full 0-255.

    Applied at BOTH train and inference so it acts as a domain-invariance
    step, not a test-time hack: dataset scans already span the full range
    (so this is close to a no-op for them), while phone photos of paper
    never reach true black/white under ambient light. Normalizing that
    away here means the model never has to learn around it.
    """
    arr = np.asarray(pil_img, dtype=np.float32)
    lo, hi = np.percentile(arr, [cutoff, 100 - cutoff])
    if hi - lo < 1.0:
        return pil_img
    out = np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def base_transform(pil_img):
    """Deterministic geometry+contrast normalization applied everywhere."""
    img = pil_img.convert("L")
    img = to_square(img, fill=255)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img = stretch_contrast(img)
    return img


def preprocess_symbol(pil_img):
    """Full inference-time preprocessing -> normalized float32 [H,W] array."""
    img = base_transform(pil_img)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - NORM_MEAN) / NORM_STD
    return arr.astype(np.float32)
