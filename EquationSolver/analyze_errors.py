"""Per-class diagnostics on the held-out test split.

Answers 'which symbols does the model actually confuse?', which raw
accuracy cannot. Runs against the same images the Kaggle notebook held
out (artifacts/test_split_manifest.txt), so it needs no retraining.

    .venv/bin/python analyze_errors.py
"""
from collections import defaultdict
from pathlib import Path

from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from classes import CLASSES, class_to_display
from pipeline import classify_crop, load_model

DATA_ROOT = Path.home() / ".cache/kagglehub/datasets/sagyamthapa/handwritten-math-symbols/versions/4/dataset"
MANIFEST = Path(__file__).parent / "artifacts" / "test_split_manifest.txt"


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"{MANIFEST} not found -- download it from the Kaggle notebook output."
        )

    model, classes = load_model()

    y_true, y_pred = [], []
    for line in MANIFEST.read_text().strip().splitlines():
        fname, cls = line.split("|")
        path = DATA_ROOT / cls / fname
        if not path.exists():
            continue
        pred, _conf = classify_crop(model, classes, Image.open(path))
        y_true.append(cls)
        y_pred.append(pred)

    labels = [c for c in CLASSES]
    display = [class_to_display(c) for c in labels]

    print(f"held-out test images: {len(y_true)}\n")
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=display, digits=3, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("\nconfusion matrix (rows = true, cols = predicted)")
    print("      " + "".join(f"{d:>5}" for d in display))
    for name, row in zip(display, cm):
        cells = "".join(f"{v:>5}" if v else "    ." for v in row)
        print(f"{name:>5} {cells}")

    # the actionable part: which specific pairs get mixed up
    pairs = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t != p:
            pairs[(t, p)] += 1

    print("\nmost common confusions:")
    if not pairs:
        print("  none -- perfect on the held-out split")
    for (t, p), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
        print(f"  {class_to_display(t)} -> {class_to_display(p)}   {n}x")


if __name__ == "__main__":
    main()
