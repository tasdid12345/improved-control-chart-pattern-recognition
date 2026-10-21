# Control Chart Pattern Recognition: 1-NN Template Matching vs Class-Balanced LightGBM

Statistical process control relies on recognising unnatural patterns in control charts, such as trends and shifts, early. The paper "Pattern matching for control chart monitoring" (2008) addresses this with pattern-matching methods. Its exact method is not specified in the abstract, so this experiment uses a standard 1-nearest-neighbour template matcher as a stand-in baseline and asks whether a gradient-boosting model on engineered features does better.

**Question:** Can a class-balanced gradient-boosting model on engineered trend and shift features beat a 1-nearest-neighbour template-matching baseline at recognising control chart patterns on the UCI Synthetic Control Chart dataset?

**Dataset:** UCI Synthetic Control Chart Time Series (600 series of length 60, 6 classes: normal, cyclic, increasing trend, decreasing trend, upward shift, downward shift), downloaded as a plain text file. This is not the paper's own dataset. A stratified 70/30 split is made with seed 42.

**Baseline:** 1-nearest-neighbour classifier with Euclidean distance on the raw series.

**Improved model:** Class-balanced LightGBM trained on the raw series plus engineered features (linear slope, segment means, shift magnitude, variance, peak-to-peak range). Hyperparameters are chosen by stratified cross-validation on the training part only. The test set is used once at the end.

**Metric:** Macro-F1 on the test set, with accuracy and per-class recall as secondary metrics.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json`; charts are saved to `figures/`.

## Reference

This project is motivated by the following paper:

> D Cantone, S Faro (2008). *Pattern matching for control chart monitoring*. Progress in Industrial Mathematics at ECMI 2006. https://link.springer.com/content/pdf/10.1007/978-3-540-71992-2_161.pdf

Paper author: Prof. Assoc. Prof. Simone Faro, University of Catania (faro@dmi.unict.it)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
