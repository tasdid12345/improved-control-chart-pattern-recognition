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

## Reference

This project is motivated by the following paper:

> D Cantone, S Faro (2008). *Pattern matching for control chart monitoring*. Progress in Industrial Mathematics at ECMI 2006. https://link.springer.com/content/pdf/10.1007/978-3-540-71992-2_161.pdf

Paper author: Prof. Assoc. Prof. Simone Faro, University of Catania (faro@dmi.unict.it)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
