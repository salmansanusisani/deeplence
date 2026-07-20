"""
Benchmark script: runs the detector against the labeled test_set/ folder
and reports real accuracy, precision, recall, and a confusion matrix.

Two sub-sets are evaluated separately because they're very different in
nature, and averaging them together would be misleading:

  1. test_set/real/ + test_set/fake/
     -> 80 samples (40 real + 40 fake) from the CIFAKE benchmark dataset
        (Bird & Lotfi, 2023, arXiv:2303.14126) — real CIFAR-10 photos vs.
        their Stable Diffusion v1.4-generated equivalents. NOTE: these are
        32x32 pixel images — far lower resolution than this tool is designed
        for (ELA/FFT/noise signals need real texture detail to work well),
        so treat this as a conservative / pessimistic lower-bound benchmark.

  2. test_set/realistic_samples/
     -> 3 full-resolution labeled images (1 real photo, 2 AI-generated),
        sourced from a public GitHub test-image set. Small sample size, but
        realistic resolution — a sanity check that the pipeline behaves
        correctly on the kind of input it will actually see at a hackathon.

Run:
    python3 benchmark.py
"""

import os
import sys
import time
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from app.detector import analyze_image

THRESHOLD = 50.0  # percent, matches the "Uncertain" cutoff used in the app


def evaluate_folder(real_dir, fake_dir, label):
    results = []
    for fname in sorted(os.listdir(real_dir)):
        path = os.path.join(real_dir, fname)
        results.append((path, "real"))
    for fname in sorted(os.listdir(fake_dir)):
        path = os.path.join(fake_dir, fname)
        results.append((path, "fake"))

    tp = fp = tn = fn = 0
    errors = 0
    abstentions = 0
    predictions = []

    for path, true_label in results:
        try:
            img = Image.open(path)
            img.load()
            raw = open(path, "rb").read()
            result = analyze_image(img, raw)
            if not result.get("decision_ready"):
                abstentions += 1
                predictions.append((os.path.basename(path), true_label, "inconclusive", result["ai_probability_percent"]))
                continue
            pct = result["ai_probability_percent"]
            predicted = "fake" if pct >= THRESHOLD else "real"
            predictions.append((os.path.basename(path), true_label, predicted, pct))

            if true_label == "fake" and predicted == "fake":
                tp += 1
            elif true_label == "real" and predicted == "fake":
                fp += 1
            elif true_label == "real" and predicted == "real":
                tn += 1
            elif true_label == "fake" and predicted == "real":
                fn += 1
        except Exception as e:
            errors += 1
            print(f"  [error] {path}: {e}")

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\n=== {label} ===")
    print(f"Decision-ready samples: {total} (inconclusive: {abstentions}, errors: {errors})")
    coverage = total / (total + abstentions) if (total + abstentions) else 0
    print(f"Decision coverage: {coverage*100:.1f}%")
    print(f"Confusion matrix:  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Accuracy:  {accuracy*100:.1f}%")
    print(f"Precision: {precision*100:.1f}%  (of images flagged AI, how many really were)")
    print(f"Recall:    {recall*100:.1f}%  (of actual AI images, how many were caught)")
    print(f"F1 score:  {f1*100:.1f}%")

    return predictions


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    test_set = os.path.join(base, "test_set")

    start = time.time()

    print("Running benchmark against CIFAKE sample (40 real + 40 fake, 32x32 native res)...")
    preds_cifake = evaluate_folder(
        os.path.join(test_set, "real"),
        os.path.join(test_set, "fake"),
        "CIFAKE sample (low-res, pessimistic lower bound)"
    )

    realistic_dir = os.path.join(test_set, "realistic_samples")
    if os.path.isdir(realistic_dir):
        print("\n\n=== Realistic-resolution sanity check ===")
        for fname in sorted(os.listdir(realistic_dir)):
            path = os.path.join(realistic_dir, fname)
            true_label = "real" if fname.startswith("real_") else "fake"
            img = Image.open(path)
            img.load()
            raw = open(path, "rb").read()
            result = analyze_image(img, raw)
            pct = result["ai_probability_percent"]
            predicted = "fake" if pct >= THRESHOLD else "real"
            correct = "✓" if predicted == true_label else "✗"
            print(f"  {correct} {fname:35s} true={true_label:5s} predicted={predicted:5s} score={pct}%")

    elapsed = time.time() - start
    print(f"\nTotal benchmark time: {elapsed:.1f}s")
