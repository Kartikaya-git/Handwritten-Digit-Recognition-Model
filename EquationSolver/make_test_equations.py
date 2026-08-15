"""Stitch held-out symbol images (per artifacts/test_split_manifest.txt) into
synthetic handwritten-equation images, for end-to-end pipeline testing.

The manifest guarantees these symbol crops were never seen during training
on Kaggle, so equation-level accuracy measured on them isn't inflated by
data leakage.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

from classes import CLASSES, class_to_display

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DATA_ROOT = Path.home() / ".cache/kagglehub/datasets/sagyamthapa/handwritten-math-symbols/versions/4/dataset"
OUT_DIR = Path(__file__).parent / "test_equations"

OPERATORS = ["add", "sub", "mul", "div"]
DIGITS = [c for c in CLASSES if c.isdigit()]


def load_manifest():
    manifest_path = ARTIFACTS_DIR / "test_split_manifest.txt"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Download test_split_manifest.txt from the "
            f"Kaggle notebook's Output tab and place it in EquationSolver/artifacts/."
        )
    by_class = defaultdict(list)
    for line in manifest_path.read_text().strip().splitlines():
        fname, cls = line.split("|")
        by_class[cls].append(DATA_ROOT / cls / fname)
    return by_class


def make_equation(by_class, rng, max_operand=99):
    a = rng.randint(0, max_operand)
    b = rng.randint(0, max_operand)
    op = rng.choice(OPERATORS)
    if op == "div":
        b = rng.randint(1, 12)
        a = b * rng.randint(0, 12)  # keep it a clean division

    symbol_seq = [d for d in str(a)] + [op] + [d for d in str(b)]
    if op == "add":
        answer = a + b
    elif op == "sub":
        answer = a - b
    elif op == "mul":
        answer = a * b
    else:
        answer = a // b

    imgs = []
    for cls in symbol_seq:
        path = rng.choice(by_class[cls])
        imgs.append(Image.open(path).convert("L"))

    gap = 12
    heights = [im.height for im in imgs]
    target_h = max(heights)
    resized = []
    for im in imgs:
        scale = target_h / im.height
        resized.append(im.resize((max(1, int(im.width * scale)), target_h)))

    total_w = sum(im.width for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("L", (total_w, target_h), color=255)
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.width + gap

    expr_display = "".join(
        class_to_display(c) if c in ("add", "sub", "mul", "div") else c for c in symbol_seq
    )
    return canvas, symbol_seq, expr_display, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=30, help="number of synthetic equations")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    by_class = load_manifest()
    rng = random.Random(args.seed)
    OUT_DIR.mkdir(exist_ok=True)

    manifest = []
    for i in range(args.n):
        canvas, symbol_seq, expr_display, answer = make_equation(by_class, rng)
        fname = f"eq_{i:03d}.png"
        canvas.save(OUT_DIR / fname)
        manifest.append({
            "file": fname,
            "expression": expr_display,
            "symbols": symbol_seq,
            "answer": answer,
        })

    (OUT_DIR / "ground_truth.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {args.n} synthetic equations to {OUT_DIR}")


if __name__ == "__main__":
    main()
