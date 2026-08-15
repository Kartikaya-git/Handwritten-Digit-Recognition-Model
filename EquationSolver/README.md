# Equation Solver

End-to-end handwritten math equation recognizer: segmentation → CNN
classification (digits 0-9 + operators + − × ÷) → rule-based parsing → solve.

Training happens on Kaggle (GPU), not locally. Everything else
(segmentation, parsing, evaluation) runs locally in `.venv`.

> **See [`PROJECT_NOTES.md`](PROJECT_NOTES.md)** for design decisions,
> what was tried and rejected, measured failure modes, honest limitations,
> and future scope (including the segmentation alternatives that were
> researched and deferred).

## Results (measured, not aspirational)

Trained on `sagyamthapa/handwritten-math-symbols` (14 classes: digits 0-9 +
add/sub/mul/div), 45 epochs on a Kaggle T4 GPU.

| Metric | v1 | v2 (current) |
|---|---|---|
| Symbol classification (Kaggle held-out test set) | 96.8% | **98.9%** |
| Digit-level accuracy (end-to-end) | 95.0% | **97.3%** |
| Symbol-level accuracy (end-to-end) | 95.7% | **97.5%** |
| Equation-level accuracy (end-to-end) | 86.7% | **93.3%** (56/60) |
| Real phone photos of handwriting | 0/3 | **3/3** |
| MNIST-style renders (`../test_images`) | 2/10 | **10/10** |

The end-to-end numbers come from `evaluate.py` run against 60 synthetic
equation images built exclusively from symbol crops the model never saw
during training (see `test_split_manifest.txt` below) — segmentation,
classification, parsing, and solving are all exercised together, so
there's no leakage inflating the numbers.

In the trained model, per-class test accuracy is 100% for all four
operators and ≥94.7% for every digit.

### Where the remaining errors are

`analyze_errors.py` runs the trained model over the held-out split and
prints a scikit-learn `classification_report` plus a confusion matrix.
Only **9 of 788** held-out symbols are misclassified, and they are not
randomly distributed:

| Confusion | Count |
|---|---|
| 0 → 6 | 2 |
| 1 → 7 | 2 |
| 0 → 8, 1 → +, 4 → +, 6 → 0, 7 → 9 | 1 each |

Two things worth noting. **All four operators are perfect** — 100%
precision *and* recall — so the residual error is entirely among digits.
And the mistakes are visually explainable rather than arbitrary: 0↔6 is
symmetric (a closed loop with or without a tail), 1→7 and 7→9 are
stroke-shape ambiguities, and 1/4 → + happens when a crossing stroke
reads as a horizontal bar.

That distinction matters for what to do next: these are inherently
ambiguous glyphs, not underfitting, so more of the same training data
would have limited returns — higher input resolution or shape-aware
features would be the lever.

### What changed between v1 and v2

v1 scored well on the benchmark but got **every real phone photo wrong,
at 100% confidence** — a textbook train/deploy distribution gap. Three
fixes, in order of impact:

1. **Photometric augmentation** (`ColorJitter`, `GaussianBlur`). v1 only
   ever saw clean, evenly-lit, sharp scans, so a dim photo of paper was
   off-distribution. This is the fix that made real photos work.
2. **Geometric augmentation** ported from `../CNN_PyTorch/train_cnn.py` —
   rotation, translation, scale, perspective, random erasing. Perspective
   matters most here: photos are rarely taken exactly square-on.
3. **Consistent normalization.** v1 had none at all. v2 measures this
   dataset's own mean/std (0.9435 / 0.2053) and applies the identical
   preprocessing at train and inference time — verified byte-identical,
   since a silent mismatch there is invisible in benchmarks but wrecks
   real predictions. (The MNIST code hardcoded MNIST's 0.1307/0.3081,
   which is inverted for this mostly-white-paper data.)

Plus `normalize_polarity()` in `segment.py`, which auto-detects
light-ink-on-dark vs dark-ink-on-light. That alone took the old
MNIST-style test images from 2/10 to 10/10.

### Augmentation

Three groups, all active during training (see `augmentations.py`):

| Group | Transforms | Why |
|---|---|---|
| Real-world damage | random occlusion, partial erasure, noise injection | smudges, cut-off strokes, sensor noise |
| Geometric | rotation ±10°, translate ±10%, scale 0.85–1.1, perspective 0.3, random erasing | ported from the MNIST CNN; handles angled/offset photos |
| Photometric | brightness ±35%, contrast ±35%, Gaussian blur | exposure and focus variation in phone photos |

Every geometric op uses `fill=255`, since this is dark ink on white paper —
torchvision's default black fill would paste fake black borders into every
rotated sample. Train accuracy (~93%) sits *below* validation accuracy
(~99%) as a result: training samples are deliberately damaged, validation
ones are clean. That gap is the augmentation working, not underfitting.

## 1. Train on Kaggle

Already run once — see Results above. `notebook/train_on_kaggle.ipynb` was
pushed and executed via the `kaggle` CLI (kernel push → poll status → pull
output), not the browser UI. To retrain (e.g. after changing the model or
augmentation):

1. Push and run:
   ```bash
   cp notebook/train_on_kaggle.ipynb kernel_push/
   cd kernel_push && ../.venv/bin/kaggle kernels push
   ../.venv/bin/kaggle kernels status kartikayakaggle/equation-symbol-cnn-train
   ```
2. Once status is `COMPLETE`, pull the outputs:
   ```bash
   ../.venv/bin/kaggle kernels output kartikayakaggle/equation-symbol-cnn-train -p ../artifacts
   ```

This downloads `equation_cnn.pth`, `classes.json`, and
`test_split_manifest.txt` straight into `EquationSolver/artifacts/`.

Or run it manually in the browser instead: Add Data → search
`sagyamthapa/handwritten-math-symbols` → Add; Settings → Accelerator → GPU;
File → Import Notebook → upload `notebook/train_on_kaggle.ipynb`; Run All;
download the same 3 files from the Output tab.

`test_split_manifest.txt` lists exactly which images the notebook held out
as its test set — the local pipeline uses only those to build synthetic
test equations, so there's no data leakage in the reported accuracy.

## 2. Generate synthetic test equations (local)

```bash
cd EquationSolver
.venv/bin/python make_test_equations.py -n 50
```

Stitches held-out digit/operator crops into equation images (e.g. `12+7`)
under `test_equations/`, with `ground_truth.json` recording the true
symbols and answer for each.

## 3. Evaluate end-to-end accuracy

```bash
.venv/bin/python evaluate.py
```

Reports digit-level, symbol-level, and equation-level accuracy by running
the full pipeline (segment → classify → parse → solve) on each synthetic
equation and comparing to ground truth.

## 4. Web UI (upload and try it)

```bash
EquationSolver/.venv/bin/python EquationSolver/app.py
```

Then open <http://127.0.0.1:5001>. Drag in a photo (or click one of the
built-in examples) and you get back:

- the answer and the reconstructed expression
- the original image with a box drawn per detected symbol, labelled with
  its prediction and confidence — **green ≥80%, amber below**, so a shaky
  read is obvious at a glance
- a per-symbol confidence table
- a collapsible "what the segmenter saw" view showing the image after
  polarity normalization — handy when a result looks wrong

Uploads are staged in a temp file and deleted immediately after inference;
nothing is retained on disk.

## 5. Try it from the command line

```bash
.venv/bin/python pipeline.py path/to/your_equation.jpg
```

Write digits/operators on plain paper (dark ink on light background),
photograph it, and pass the path in. Prints each detected symbol with
confidence, the reconstructed expression, and the answer.

## Files

- `model_cnn_def.py` — `EquationCNN` architecture (3 conv blocks + FC)
- `augmentations.py` — the three augmentation groups (see below)
- `preprocessing.py` — shared train/inference preprocessing + norm stats
- `segment.py` — OpenCV contour-based symbol segmentation: polarity
  normalization → Otsu threshold → dilate → `findContours` → bounding
  boxes → merge same-column boxes → sort left-to-right
- `solver.py` — symbols → expression string → safe AST-based evaluation
- `pipeline.py` — wires segmentation + model + solver together
- `app.py` + `templates/index.html` — the local web UI
- `make_test_equations.py` / `evaluate.py` — leak-free end-to-end testing
- `analyze_errors.py` — per-class report + confusion matrix (scikit-learn)
- `notebook/train_on_kaggle.ipynb` — self-contained training notebook
