import json
import sys
from pathlib import Path

import numpy as np
import torch

from classes import CLASSES
from model_cnn_def import EquationCNN
from preprocessing import preprocess_symbol
from segment import segment_equation
from solver import ParseError, solve_symbols

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def load_model(weights_path=None, classes_path=None):
    weights_path = weights_path or ARTIFACTS_DIR / "equation_cnn.pth"
    classes_path = classes_path or ARTIFACTS_DIR / "classes.json"

    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"{weights_path} not found. Train on Kaggle (notebook/train_on_kaggle.ipynb), "
            f"download equation_cnn.pth + classes.json from the Output tab, and place them here."
        )

    classes = json.loads(Path(classes_path).read_text()) if Path(classes_path).exists() else CLASSES

    model = EquationCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model, classes


def classify_crop(model, classes, pil_crop):
    arr = preprocess_symbol(pil_crop)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx])


def solve_image(image_path, model=None, classes=None, verbose=False):
    if model is None or classes is None:
        model, classes = load_model()

    crops = segment_equation(image_path)
    if not crops:
        raise ParseError(f"no symbols found in {image_path}")

    predicted = []
    confidences = []
    boxes = []
    for crop, bbox in crops:
        cls, conf = classify_crop(model, classes, crop)
        predicted.append(cls)
        confidences.append(conf)
        boxes.append(bbox)
        if verbose:
            print(f"  bbox={bbox}  ->  {cls}  (conf={conf:.3f})")

    expr, result = solve_symbols(predicted)
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return {
        "expression": expr,
        "result": result,
        "symbols": predicted,
        "confidences": confidences,
        "boxes": boxes,
        "avg_confidence": avg_conf,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python pipeline.py <image_path> [image_path ...]")
        sys.exit(1)

    model, classes = load_model()
    for path in sys.argv[1:]:
        try:
            out = solve_image(path, model=model, classes=classes, verbose=True)
            print(f"{path}: {out['expression']} = {out['result']}  "
                  f"(avg conf {out['avg_confidence']:.3f})")
        except ParseError as e:
            print(f"{path}: FAILED - {e}")
