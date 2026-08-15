"""End-to-end evaluation: segmentation + classification + parsing + solving,
run against synthetic equations built from held-out (never-trained-on) crops.

Reports digit-level accuracy, full symbol-level accuracy (digits+operators),
and equation-level accuracy (segmentation count matches AND final answer correct).
"""
import json
from pathlib import Path

from pipeline import load_model, solve_image
from solver import ParseError

TEST_DIR = Path(__file__).parent / "test_equations"


def main():
    gt_path = TEST_DIR / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"{gt_path} not found. Run make_test_equations.py first.")

    ground_truth = json.loads(gt_path.read_text())
    model, classes = load_model()

    digit_correct = digit_total = 0
    symbol_correct = symbol_total = 0
    eq_correct = 0
    failures = []

    for item in ground_truth:
        img_path = TEST_DIR / item["file"]
        gt_symbols = item["symbols"]
        try:
            out = solve_image(img_path, model=model, classes=classes)
        except ParseError as e:
            failures.append((item["file"], f"pipeline error: {e}"))
            continue

        pred_symbols = out["symbols"]

        if len(pred_symbols) == len(gt_symbols):
            for p, g in zip(pred_symbols, gt_symbols):
                symbol_total += 1
                if p == g:
                    symbol_correct += 1
                if g.isdigit():
                    digit_total += 1
                    if p == g:
                        digit_correct += 1
        else:
            symbol_total += len(gt_symbols)
            digit_total += sum(1 for g in gt_symbols if g.isdigit())
            failures.append((item["file"], f"segmentation count mismatch: "
                                            f"got {len(pred_symbols)}, expected {len(gt_symbols)}"))

        if out["result"] == item["answer"]:
            eq_correct += 1
        else:
            failures.append((item["file"], f"wrong answer: predicted {out['expression']}="
                                            f"{out['result']}, expected {item['expression']}={item['answer']}"))

    n = len(ground_truth)
    print(f"equations tested: {n}")
    print(f"digit-level accuracy:     {digit_correct}/{digit_total} = {digit_correct/max(digit_total,1):.3%}")
    print(f"symbol-level accuracy:    {symbol_correct}/{symbol_total} = {symbol_correct/max(symbol_total,1):.3%}")
    print(f"equation-level accuracy:  {eq_correct}/{n} = {eq_correct/max(n,1):.3%}")

    if failures:
        print(f"\n{len(failures)} failures:")
        for fname, reason in failures[:20]:
            print(f"  {fname}: {reason}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")


if __name__ == "__main__":
    main()
