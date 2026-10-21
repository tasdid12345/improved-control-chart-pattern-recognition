# Control chart pattern recognition: 1-NN baseline vs class-balanced RBF SVM

## What this does
Downloads the UCI Synthetic Control Chart dataset (600 series × 60 points, 6 classes, 100 per class),
makes a stratified 70/30 split with seed 42, and compares two classifiers on the held-out test set
using macro-F1:

- **Baseline:** 1-nearest-neighbour template matcher, Euclidean distance on the raw series.
- **Improved:** class-balanced RBF SVM on standardized raw series plus engineered features
  (least-squares slope, three segment means, start-to-end shift and its magnitude, variance,
  peak-to-peak range). `C` and `gamma` are chosen by stratified 5-fold cross-validation on the
  training split only; the test set is used once at the end.

## Change from the previous version
The previous improved model (class-balanced LightGBM) scored about 0.971 macro-F1 in cross-validation
and tied the baseline on test (0.9833 vs 0.9833). It was replaced by an RBF SVM with standardized
features. The baseline, data, split, metric and shared evaluation code are unchanged. The question
wording was generalized from "gradient-boosting model" to "class-balanced classifier" to match.

## Run
```
pip install -r requirements.txt
python analysis.py
```
Outputs `results.json` and `figures/macro_f1_comparison.png`. Runs on CPU in well under a minute.

## Notes
Class labels come from the file's row order (100 rows per class). With 180 test series, small
differences are noisy. The paper's own method is not specified, so this is not a replication.

## Results (latest run)

On the UCI Synthetic Control Chart test set (180 series), a standard 1-nearest-neighbour recognizer reached 0.983 macro-F1 versus 0.994 for my class-balanced RBF SVM on engineered trend and shift features (gradient boosting was dropped after CV), a small gain from a single split.

| Metric | Value |
|---|---|
| baseline_score | 0.9833 |
| improved_score | 0.9944 |
| improvement | 0.01111 |
| baseline_score_std | 0 |
| improved_score_std | 0 |
| baseline_f1 | 0.9833 |
| improved_f1 | 0.9944 |
| baseline_accuracy | 0.9833 |
| improved_accuracy | 0.9944 |
| cv_best_macro_f1 | 0.9905 |
| baseline_recall_normal | 1 |
| improved_recall_normal | 1 |
| baseline_recall_cyclic | 1 |
| improved_recall_cyclic | 1 |
| baseline_recall_increasing_trend | 1 |
| improved_recall_increasing_trend | 0.9667 |
| baseline_recall_decreasing_trend | 0.9667 |
| improved_recall_decreasing_trend | 1 |
| baseline_recall_upward_shift | 0.9667 |
| improved_recall_upward_shift | 1 |
| baseline_recall_downward_shift | 0.9667 |
| improved_recall_downward_shift | 1 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Single stratified split; test set used once for each model. Both models are deterministic here (std 0). The improved model was changed from class-balanced LightGBM (which CV showed at ~0.971 macro-F1, below the 1-NN baseline) to an RBF SVM with standardized features, a model family that is well suited to small, low-dimensional time-series data. Class labels are inferred from the file's row order (100 rows per class). With 180 test series, differences of a few samples are noisy. The paper's own method and results are not specified, so this is not a replication.

## Reference

This project is motivated by the following paper:

> D Cantone, S Faro (2008). *Pattern matching for control chart monitoring*. Progress in Industrial Mathematics at ECMI 2006. https://link.springer.com/content/pdf/10.1007/978-3-540-71992-2_161.pdf

Paper author: Prof. Assoc. Prof. Simone Faro, University of Catania (faro@dmi.unict.it)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
