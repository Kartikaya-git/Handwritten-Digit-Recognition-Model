# Project notes — context, decisions, and future scope

Companion to `README.md`. The README says what the project does and how to
run it; this file records *why* it is built the way it is, what was tried
and rejected, where it fails, and what would come next.

---

## 1. What this is

An end-to-end handwritten arithmetic recogniser: photo of a handwritten
sum in, computed answer out. Four stages, run in sequence:

| Stage | File | Job |
|---|---|---|
| Segmentation | `segment.py` | split the image into one crop per symbol |
| Classification | `model_cnn_def.py` + trained weights | label each crop as one of 14 classes |
| Parsing | `solver.py` | assemble labels into an expression string |
| Evaluation | `solver.py` | compute the result safely |

Supported symbol set: digits `0-9` and operators `+ − × ÷` (14 classes).

The pipeline is strictly sequential, which is the single most important
property to understand: **an error at an early stage cannot be recovered
later.** A bad crop reaches the CNN as a legitimate-looking input, and the
CNN has no mechanism to signal "this isn't a symbol at all."

---

## 2. Data

- **Source:** Kaggle `sagyamthapa/handwritten-math-symbols`
- **Size used:** 7,750 images across the 14 classes of interest (~550 each)
- The dataset ships 19 classes; `dec`, `eq`, `x`, `y`, `z` are unused
- **Split:** stratified 80/10/10 → 6,194 train / 769 val / 788 test
- Splits are seeded and file lists sorted, so the split is reproducible
- The held-out test file list is written to
  `artifacts/test_split_manifest.txt` so downstream end-to-end evaluation
  can be built *only* from images the model never trained on

**Gotcha:** the dataset contains stray `.directory` files that are not
images and crash PIL. The loader filters on file extension.

**Training:** 45 epochs on a Kaggle T4 GPU. Pushed and executed through
the `kaggle` CLI (kernel push → poll status → pull artifacts), not the
browser UI.

---

## 3. Results

### Symbol classification (Kaggle held-out test split, 788 images)

| | v1 | v2 (current) |
|---|---|---|
| Overall accuracy | 96.8% | **98.9%** |

Per class in v2: **100% for all four operators**; every digit ≥ 94.7%.

### End-to-end (60 synthetic equations, held-out symbols only)

| Metric | v1 | v2 |
|---|---|---|
| Digit-level | 95.0% | **97.3%** |
| Symbol-level | 95.7% | **97.5%** |
| Equation-level | 86.7% | **93.3%** (56/60) |

### Out-of-distribution checks

| Input type | v1 | v2 |
|---|---|---|
| Real phone photos of handwriting | 0/3 | **3/3** |
| MNIST-style renders (`../test_images`) | 2/10 | **10/10** |

Equation-level counts an equation correct only if the final computed
answer matches, so it compounds every stage.

---

## 4. Why equation-level accuracy is lower than symbol-level

This is structural, not a defect, and it is worth being able to derive on
the spot.

An equation is correct only if **every** symbol in it is correct. With
per-symbol accuracy `p` and `n` symbols, accuracy is roughly `p^n`.

Measured: `p = 0.9748`, mean 4.72 symbols per equation.

```
0.9748 ^ 4.72  ≈  0.887   predicted
                   0.933   observed
```

So compounding alone explains the gap. Observed comes out slightly better
than the independence model predicts because errors cluster — a single bad
equation often absorbs two mistakes rather than spoiling two equations.

Per-length: `p^3 ≈ 0.926`, `p^4 ≈ 0.903`, `p^5 ≈ 0.880`. Longer sums are
strictly harder, and no amount of tuning removes that; only raising `p`
does.

### Failure breakdown (the 4 failing equations)

| Cause | Count |
|---|---|
| Classification error | 2 |
| Segmentation error | 2 |

**Important nuance:** neither segmentation failure is caused by adjacent
symbols overlapping or touching. Both are the *opposite* problem — one
symbol **fragmenting** into multiple boxes:

- a stray ~10px speck detected as an extra symbol, turning `46` into `146`
- a digit split into two contours that the merge rule missed by a hair
  (overlap ratio 0.486 against a 0.50 threshold)

This distinction matters because the fixes are opposite: fragmentation
needs better merging or small-blob filtering; genuine overlap needs better
splitting.

**Caveat on the benchmark:** synthetic equations are stitched with a fixed
gap, so adjacent symbols *cannot* touch. Real cramped handwriting would
introduce genuine overlap, which this benchmark does not measure. Claims
about overlap robustness are therefore unsupported by current evidence.

### Where the classifier still errs

From `analyze_errors.py` (scikit-learn `classification_report` +
`confusion_matrix`): **9 errors in 788 symbols**, and they are systematic
rather than random.

| Confusion | Count |
|---|---|
| 0 → 6 | 2 |
| 1 → 7 | 2 |
| 0 → 8, 1 → +, 4 → +, 6 → 0, 7 → 9 | 1 each |

All operators are perfect on both precision and recall, so residual error
is entirely among digits, and the confusions are visually explainable:
0↔6 is symmetric (closed loop ± tail), 1→7 and 7→9 are stroke-shape
ambiguities, 1/4 → + is a crossing stroke reading as a bar.

**Implication:** these are inherent glyph ambiguities, not underfitting.
More of the same training data would give poor returns; higher input
resolution or shape-aware features are the real lever.

---

## 5. Key engineering decisions

### Segmentation: contour bounding boxes, with column-based merging

`findContours` returns one contour per connected ink region. That breaks
on multi-stroke symbols: `÷` is three disconnected pieces (dot, bar, dot),
so a naive "one contour = one symbol" rule over-counts and then tries to
classify a lone dot, which is not any of the 14 classes.

**Rule used:** merge boxes that substantially share a *horizontal*
footprint — i.e. are stacked in the same column. The parts of `÷` all
occupy the same vertical strip; adjacent digits do not.

Two details that make it work:

1. The overlap is normalised by the **smaller** of the two widths. A small
   dot fully inside a wide bar then scores 1.0 and always merges.
   Normalising by the larger width would score ~0.37 and fail.
2. Merging is **recursive** — gluing bar+dot creates a taller box that may
   then overlap something else, so it repeats until the count is stable.

**Rejected alternative:** merging by *proximity* (boxes close together).
This was the first implementation and it was wrong — it glued adjacent
digits in multi-digit numbers into one blob. Equation-level accuracy was
66.7%; switching to pure column-overlap took it to 86.7% in one change.

**Threshold tuning:** swept the overlap threshold across 0.35–0.60 and
measured no change in equation-level accuracy or segment-count accuracy at
any value. The threshold is therefore not a sensitive parameter on this
data, and was left at 0.50.

**Known limitation:** the rule assumes multi-part symbols stack
*vertically*. It handles `÷`, `=`, `i`, `%`. It would fail for a symbol
whose parts sit side by side, and would wrongly merge two digits written
cramped enough to overlap horizontally.

### Polarity normalisation

Inputs arrive both as dark-ink-on-white-paper (photos) and
light-ink-on-dark (MNIST-style renders). `normalize_polarity()` converts
everything to the canonical dark-on-light form.

Decided by **which Otsu class is the minority**, on the reasoning that ink
always covers less of the page than background does, under any lighting.

Two simpler rules were tried and both failed on a real vignetted phone
photo whose paper corners were darker than its ink:

- absolute brightness cutoff — the border sat exactly on the threshold
- "which class does the border fall in" — the vignette pushed the border
  into the dark class, so plain white paper read as light-on-dark

This single function took the older MNIST-style test images from 2/10 to
10/10.

### Safe evaluation instead of `eval()`

`solver.py` parses the expression to an AST and walks it, permitting only
the four arithmetic operations and numeric literals. Using `eval()` on a
string assembled from model predictions would be arbitrary code execution
driven by image content.

Consecutive digits are grouped and passed through `int()` so leading zeros
(e.g. a predicted `05`) do not break Python's literal grammar — a real bug
found during evaluation.

### Shared preprocessing between training and inference

Training and inference preprocessing live in one place (`preprocessing.py`,
duplicated verbatim into the notebook) and were verified to produce
byte-identical output (max absolute difference 0.0).

This is deliberate: a train/inference preprocessing mismatch is invisible
in benchmark numbers — the benchmark uses the training preprocessing — but
silently wrecks real-world predictions.

Normalisation uses this dataset's own measured statistics
(**mean 0.9435, std 0.2053**). The earlier MNIST-era code hardcoded MNIST's
`(0.1307, 0.3081)`, which is effectively inverted for this data: these
images are mostly white paper, so the mean is high.

---

## 6. The distribution-gap debugging story

The most instructive episode in the project, and the strongest interview
material.

**Symptom:** v1 scored 96.8% on its benchmark but got **every real phone
photo wrong — at 100% confidence.**

**Diagnosis:** the Kaggle images span the full black-to-white range. Real
photos of paper, under ambient light, never reach true black or true
white; one test photo's ink never got darker than mid-grey. The CNN had
never seen a washed-out input, and BatchNorm-based CNNs tend to be
confidently wrong on out-of-distribution input rather than uncertain.

**What did not work:** hard binarisation of each crop. It fixed the failing
photo (99.7% correct) but dropped benchmark equation accuracy from 86.7%
to 71.7%, because the model was trained on soft, anti-aliased greyscale,
not hard black-and-white. Reverted rather than special-casing real photos,
since that would paper over the gap instead of closing it.

**What worked** — three changes, in order of impact:

1. **Photometric augmentation** (`ColorJitter`, `GaussianBlur`). v1 had
   none, because it inherited its augmentation from an MNIST project where
   the data is already clean and uniform. This is the change that made real
   photos work.
2. **Geometric augmentation** — rotation, translation, scale, perspective,
   random erasing, ported from `../CNN_PyTorch/train_cnn.py`. Perspective
   matters most: photos are rarely taken exactly square-on.
3. **Consistent normalisation** with dataset-specific statistics.

Every geometric op uses `fill=255`, since this is dark ink on white paper;
torchvision's default black fill would paste fake black borders into every
rotated sample.

**Diagnostic worth keeping:** in v2, train accuracy (~93%) sits *below*
validation accuracy (~99%). That inversion is the augmentation working —
training samples are deliberately damaged, validation samples are clean —
not underfitting.

**Tuning note:** the first augmentation configuration was too aggressive.
Three overlapping "remove content" transforms (occlusion, partial erasure,
random erasing) stacked into unrecognisable samples. Probabilities and
magnitudes were reduced after inspecting a grid of augmented samples, which
the notebook still renders as a sanity check.

---

## 7. Honest limitations

- **Flat, single-line expressions only.** No fractions, exponents, roots,
  parentheses, decimals, or `=`. The parser is left-to-right with no
  concept of 2D layout.
- **No variables or equation solving** in the algebraic sense — this
  evaluates arithmetic, it does not solve for `x`.
- **Equation-level accuracy is measured on synthetic images** stitched from
  real symbol crops, not on photographs of complete handwritten sums. The
  symbol images are real and held-out, but the spacing and layout are
  artificial and forgiving.
- **Real-photo evidence is thin** — 3 photos of single digits, not a
  measured set of photographed equations. The correct claim is "fixed a
  reproducible failure mode", not "X% accurate on real photos".
- Segmentation assumes symbols are separated by whitespace and written on
  one horizontal line.
- No confidence-based rejection: the pipeline always returns an answer,
  even when every symbol is low-confidence. The UI surfaces confidence, but
  nothing acts on it.

---

## 8. Future scope

### Segmentation alternatives (researched, deliberately deferred)

| Approach | How it works | Trade-off |
|---|---|---|
| **Contour bounding boxes** (current) | threshold → contours → boxes → merge → classify each | Simple, fast, well-proven for flat single-line sums. Breaks on touching symbols and multi-part glyphs without explicit merge rules. |
| **CRNN + CTC** | treat the whole line as a sequence; no explicit per-symbol segmentation | Handles touching/overlapping symbols naturally and removes the fragile merge heuristics entirely. Needs synthetic *sequence* training data (stitched equation images) and CTC loss — meaningfully more engineering. **The most sensible next step.** |
| **YOLO-style symbol detector** | object detection over symbols | Learns segmentation instead of hand-tuning it; needs box-annotated training data, which this dataset does not provide. |
| **Attention encoder–decoder → LaTeX** | image → sequence model → LaTeX string, handles 2D layout | What research systems (CROHME) use. Necessary for fractions/exponents, overkill for flat sums. For calibration: published CROHME results report roughly ~50–53% *expression*-level accuracy, because the task includes full 2D structure — far harder than this project's scope. |

**Why bounding boxes were chosen:** the target input is flat, left-to-right
arithmetic with spacing between symbols. Contour segmentation is proven for
exactly that case, is fast to build and debug, and made an end-to-end
system reachable. The known failure modes are now measured rather than
assumed.

**Strongest argument for CRNN+CTC as v3:** two of the four current
end-to-end failures are segmentation errors, and both are the kind
(fragmentation, spurious blobs) that a sequence model sidesteps entirely
because it never has to commit to symbol boundaries.

### Other next steps

- Photograph and label a real set of complete handwritten equations, to
  replace the synthetic end-to-end benchmark with a genuine one.
- Confidence-based rejection: refuse to answer, or flag for review, when
  any symbol falls below a threshold, instead of always answering.
- Extend the symbol set to `=`, decimals, and parentheses (present in the
  dataset for `dec`/`eq`) and upgrade the parser to handle precedence and
  nesting properly.
- Higher input resolution or shape-aware features, targeting the specific
  0↔6 / 1↔7 confusions identified above.

---

## 9. Reproducing the numbers

```bash
# per-class report + confusion matrix on the held-out split
.venv/bin/python analyze_errors.py

# end-to-end: build synthetic equations from held-out symbols, then score
.venv/bin/python make_test_equations.py -n 60
.venv/bin/python evaluate.py

# single image
.venv/bin/python pipeline.py <path>

# web UI
.venv/bin/python app.py     # http://127.0.0.1:5001
```

Retraining instructions are in `README.md`. Training artifacts and the
full Kaggle training log are in `artifacts/`.
