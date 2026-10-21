"""Baseline 1-NN template matcher vs class-balanced LightGBM on engineered
trend/shift features, for control chart pattern recognition (UCI Synthetic Control)."""
import io, json, os, sys, time
import numpy as np
import requests
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier

SEED = 42
SEEDS = [42, 43, 44]  # LightGBM is randomized: run each seed, report mean/std
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data"
CLASSES = ["normal", "cyclic", "increasing_trend", "decreasing_trend", "upward_shift", "downward_shift"]


def download(url, retries=5):
    """GET with timeout=60; retry with exponential backoff on 429/5xx/network errors."""
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r.text
            if r.status_code != 429 and r.status_code < 500:
                raise RuntimeError(f"HTTP {r.status_code}")  # non-retryable client error
            msg = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            msg = str(e)
        if attempt < retries - 1:
            wait = 2 ** attempt
            print(f"  download failed ({msg}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("download failed after all retries")


def features(X):
    """Raw series plus engineered trend / shift / shape features (row-wise)."""
    n, T = X.shape
    slope = np.polyfit(np.arange(T), X.T, 1)[0]  # least-squares linear trend per series
    seg = X.reshape(n, 3, T // 3).mean(axis=2)    # means of 3 equal segments
    shift = seg[:, 2] - seg[:, 0]                 # level change start -> end (shift magnitude)
    var = X.var(axis=1)
    ptp = X.max(axis=1) - X.min(axis=1)           # peak-to-peak range
    return np.column_stack([X, slope, seg, shift, np.abs(shift), var, ptp])


def main():
    print("1/5 downloading UCI Synthetic Control data")
    X = np.loadtxt(io.StringIO(download(URL)))  # whitespace-separated, 600 x 60
    if X.shape != (600, 60):
        print(f"unexpected shape {X.shape}")
        sys.exit(1)
    # File is ordered by class: 100 consecutive rows per class (per UCI documentation)
    y = np.repeat(np.arange(len(CLASSES)), 100)

    print("2/5 stratified 70/30 split (seed 42); test set used only at the end")
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=SEED)

    def evaluate(y_true, y_pred):
        return {"f1": f1_score(y_true, y_pred, average="macro"),
                "acc": accuracy_score(y_true, y_pred),
                "recall": recall_score(y_true, y_pred, average=None, labels=range(len(CLASSES)))}

    print("3/5 baseline: 1-NN template matcher on raw series (Euclidean)")
    knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean").fit(X[tr], y[tr])
    base = evaluate(y[te], knn.predict(X[te]))

    print("4/5 improved: class-balanced LightGBM on raw + engineered features")
    F = features(X)
    Ftr, ytr, Fte = F[tr], y[tr], F[te]

    def make(params, seed):
        return lgb.LGBMClassifier(n_estimators=200, class_weight="balanced",
                                  random_state=seed, verbose=-1, **params)

    # Small grid, selected ONLY by stratified 5-fold CV on the training part
    grid = [{"num_leaves": nl, "learning_rate": lr} for nl in (7, 15, 31) for lr in (0.05, 0.1)]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = []
    for params in grid:
        fold_f1 = []
        for a, b in skf.split(Ftr, ytr):
            m = make(params, SEED).fit(Ftr[a], ytr[a])
            fold_f1.append(f1_score(ytr[b], m.predict(Ftr[b]), average="macro"))
        cv_scores.append(np.mean(fold_f1))
        print(f"  CV {params}: macro-F1={cv_scores[-1]:.4f}")
    best = grid[int(np.argmax(cv_scores))]
    print(f"  selected by CV on training data: {best}")

    runs = [evaluate(y[te], make(best, s).fit(Ftr, ytr).predict(Fte)) for s in SEEDS]
    for s, r in zip(SEEDS, runs):
        print(f"  seed {s}: test macro-F1={r['f1']:.4f}")

    print("5/5 writing results.json and chart")
    b = float(base["f1"])
    imp_f1 = [float(r["f1"]) for r in runs]
    i = float(np.mean(imp_f1))
    if not (np.isfinite(b) and np.isfinite(i)):
        sys.exit(1)

    metrics = {
        "baseline_score": b, "improved_score": i, "improvement": i - b,
        "baseline_score_std": 0.0,  # 1-NN is deterministic
        "improved_score_std": float(np.std(imp_f1)),
        "baseline_f1": b, "improved_f1": i,
        "baseline_accuracy": float(base["acc"]),
        "improved_accuracy": float(np.mean([r["acc"] for r in runs])),
    }
    for k, c in enumerate(CLASSES):
        metrics[f"baseline_recall_{c}"] = float(base["recall"][k])
        metrics[f"improved_recall_{c}"] = float(np.mean([r["recall"][k] for r in runs]))

    results = {
        "question": "Can a class-balanced gradient-boosting model on engineered trend and shift "
                    "features beat a 1-nearest-neighbour template-matching baseline at recognising "
                    "control chart patterns on the UCI Synthetic Control Chart dataset?",
        "method": "Stratified 70/30 split (seed 42); baseline 1-NN Euclidean on raw series; "
                  "improved class-balanced LightGBM on raw + engineered features with CV-selected "
                  "hyperparameters; macro-F1 on test, averaged over seeds 42-44 for LightGBM.",
        "dataset": {
            "name": "UCI Synthetic Control Chart Time Series",
            "source_url": URL,
            "n_rows": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro_f1",
        "higher_is_better": True,
        "baseline": "1-nearest-neighbour template matcher, Euclidean distance on raw 60-point series "
                    "(stand-in for the paper's unstated pattern-matching method)",
        "improved": "class-balanced LightGBM on raw series plus engineered linear slope, segment means, "
                    "shift magnitude, variance and peak-to-peak range; hyperparameters chosen by "
                    "stratified 5-fold CV on training data only",
        "metrics": metrics,
        "notes": "Single stratified split; test set used once for each model. 1-NN is deterministic "
                 "(std 0); LightGBM std is across seeds 42-44 on the same split. Class labels are "
                 "inferred from the file's row order (100 rows per class). Data is small (600 rows), "
                 "so no subsampling was needed. With 180 test series, differences are noisy. "
                 "The paper's own method and results are not specified, so this is not a replication.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline 1-NN", "improved LightGBM"], [b, i],
           yerr=[0.0, metrics["improved_score_std"]], color=["grey", "steelblue"], capsize=5)
    ax.set_ylabel("test macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Control chart pattern recognition")
    fig.tight_layout()
    fig.savefig("figures/macro_f1_comparison.png", dpi=120)
    print(f"done: baseline macro-F1={b:.4f}, improved macro-F1={i:.4f}, improvement={i - b:+.4f}")


if __name__ == "__main__":
    main()
