import os

import cv2
import numpy as np
from PIL import Image


class UnreadableImageError(ValueError):
    """Raised when a file exists but is not a decodable image."""


def _merge_overlapping(boxes, x_overlap_frac=0.5):
    """Merge bounding boxes that substantially share the same horizontal
    footprint (handles multi-stroke symbols like the two dots+bar of '/',
    which decompose into separate contours stacked in the same column).

    Deliberately does NOT merge boxes just because they're close together
    horizontally -- that's normal spacing between adjacent digits/symbols
    and must stay split."""
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = []
    for box in boxes:
        x, y, w, h = box
        placed = False
        for i, (mx, my, mw, mh) in enumerate(merged):
            ax1, ax2 = x, x + w
            bx1, bx2 = mx, mx + mw
            overlap_x = min(ax2, bx2) - max(ax1, bx1)
            min_w = min(w, mw)
            if min_w > 0 and overlap_x / min_w >= x_overlap_frac:
                nx1, ny1 = min(x, mx), min(y, my)
                nx2, ny2 = max(x + w, mx + mw), max(y + h, my + mh)
                merged[i] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                placed = True
                break
        if not placed:
            merged.append(box)
    if len(merged) != len(boxes):
        return _merge_overlapping(merged)
    return merged


def normalize_polarity(img):
    """Return the image in canonical dark-ink-on-white-paper form.

    Photos of paper are dark-on-light, but MNIST-style renders (and the
    older test images in ../test_images) are light-on-dark.

    Decided by which Otsu class is the *minority*: ink always covers less
    of the page than background does, whatever the lighting. Two simpler
    rules were tried and both failed on a real vignetted phone photo,
    where the corners of the paper are darker than the ink in the centre:
    an absolute brightness cutoff (the border sat exactly on it), and
    asking which class the border falls in (the vignette pushed the border
    into the 'dark' class, so a plain white page read as light-on-dark).
    """
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    light_fraction = (otsu == 255).mean()
    if light_fraction < 0.5:  # light class is the minority -> light ink
        return 255 - img
    return img


def segment_equation(image_path, min_area=25, pad_frac=0.15):
    """Split a handwritten equation image into an ordered list of
    (crop, bbox) for each symbol, left to right."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # cv2 returns None both for a missing file and for a file it cannot
        # decode; tell those apart so callers can report something useful.
        if not os.path.exists(str(image_path)):
            raise FileNotFoundError(image_path)
        raise UnreadableImageError(f"could not decode image: {image_path}")

    img = normalize_polarity(img)

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= min_area]
    if not boxes:
        return []

    boxes = _merge_overlapping(boxes)
    boxes.sort(key=lambda b: b[0])

    h_img, w_img = img.shape
    crops = []
    # crop from the polarity-normalized grayscale, not the raw file, so
    # light-on-dark inputs reach the model in the form it was trained on
    pil_img = Image.fromarray(img)
    for (x, y, w, h) in boxes:
        pad_x = int(w * pad_frac)
        pad_y = int(h * pad_frac)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w_img, x + w + pad_x)
        y1 = min(h_img, y + h + pad_y)
        crop = pil_img.crop((x0, y0, x1, y1))
        crops.append((crop, (x0, y0, x1, y1)))

    return crops
