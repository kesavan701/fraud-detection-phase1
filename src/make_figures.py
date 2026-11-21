"""Generate the Chapter 3 figures (3.1 - 3.10) of the Phase I report.

Usage:
    python src/make_figures.py [--data data/creditcard.csv] [--out figures]
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay)
from sklearn.model_selection import train_test_split

SEED = 42
sns.set_theme(style="whitegrid")


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / name, dpi=200)
    plt.close(fig)
    print("saved", out / name)


def main(path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)
    df["Hour"] = (df.Time // 3600) % 24
    fraud, legit = df[df.Class == 1], df[df.Class == 0]
    features = [c for c in df.columns if c not in ("Class", "Hour")]

    # Fig 3.1 - class distribution (log scale)
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df.Class.value_counts().sort_index()
    ax.bar(["Legitimate", "Fraud"], counts.values, color=["#4C72B0", "#C44E52"])
    ax.set_yscale("log")
    ax.set_ylabel("Transactions (log scale)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    save(fig, out, "fig3_1_class_distribution.png")

    # Fig 3.2 - amount distribution by class (log axis)
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.logspace(-2, np.log10(df.Amount.max() + 1), 50)
    ax.hist(legit.Amount.clip(lower=0.01), bins=bins, alpha=0.6,
            density=True, label="Legitimate")
    ax.hist(fraud.Amount.clip(lower=0.01), bins=bins, alpha=0.6,
            density=True, label="Fraud")
    ax.set_xscale("log")
    ax.set_xlabel("Amount (log scale)")
    ax.set_ylabel("Density")
    ax.legend()
    save(fig, out, "fig3_2_amount_by_class.png")

    # Fig 3.3 - volume and fraud rate by relative hour
    hourly = df.groupby("Hour").agg(n=("Class", "size"), frauds=("Class", "sum"))
    hourly["rate"] = hourly.frauds / hourly.n * 100
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(hourly.index, hourly.n, color="#4C72B0", alpha=0.7)
    ax1.set_xlabel("Relative hour")
    ax1.set_ylabel("Transactions")
    ax2 = ax1.twinx()
    ax2.plot(hourly.index, hourly.rate, color="#C44E52", marker="o")
    ax2.set_ylabel("Fraud rate (%)")
    ax2.grid(False)
    save(fig, out, "fig3_3_hourly_volume_fraud_rate.png")

    # Fig 3.4 - KS separability
    ks = pd.Series({f: ks_2samp(fraud[f], legit[f]).statistic
                    for f in features}).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#C44E52" if v > 0.6 else "#8C8C8C" for v in ks.values]
    ax.barh(ks.index[::-1], ks.values[::-1], color=colors[::-1])
    ax.set_xlabel("KS statistic")
    save(fig, out, "fig3_4_ks_separability.png")

    # Fig 3.5 - four most separable features by class
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, f in zip(axes.ravel(), ks.index[:4]):
        sns.kdeplot(legit[f], ax=ax, label="Legitimate", fill=True,
                    common_norm=False)
        sns.kdeplot(fraud[f], ax=ax, label="Fraud", fill=True,
                    common_norm=False)
        ax.set_title(f)
        ax.legend()
    save(fig, out, "fig3_5_top4_features.png")

    # Fig 3.6 - correlation matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[features].corr(), cmap="coolwarm", center=0, ax=ax)
    save(fig, out, "fig3_6_correlation_matrix.png")

    # Models for Figs 3.7 - 3.10
    X, y = df[features], df.Class
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)
    threshold = X_tr.Amount[y_tr == 1].median()
    rule_pred = (X_te.Amount > threshold).astype(int)
    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
    proba = rf.predict_proba(X_te)[:, 1]
    rf_pred = (proba >= 0.5).astype(int)

    # Fig 3.7 - confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_te, rule_pred, ax=axes[0], display_labels=["Legit", "Fraud"],
        values_format="d", colorbar=False)
    axes[0].set_title(f"Rule-based (Amount > {threshold:.2f})")
    ConfusionMatrixDisplay.from_predictions(
        y_te, rf_pred, ax=axes[1], display_labels=["Legit", "Fraud"],
        values_format="d", colorbar=False)
    axes[1].set_title("Preliminary Random Forest")
    for a in axes:
        a.grid(False)
    save(fig, out, "fig3_7_confusion_matrices.png")

    # Fig 3.8 - PR and ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    PrecisionRecallDisplay.from_predictions(y_te, proba, ax=axes[0])
    axes[0].axhline(y_te.mean(), ls="--", color="grey", label="Prevalence")
    axes[0].legend()
    axes[0].set_title("Precision-Recall")
    RocCurveDisplay.from_predictions(y_te, proba, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], ls="--", color="grey")
    axes[1].set_title("ROC")
    save(fig, out, "fig3_8_pr_roc_curves.png")

    # Fig 3.9 - top-10 feature importances
    imp = (pd.Series(rf.feature_importances_, index=features)
             .sort_values(ascending=False).head(10))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(imp.index[::-1], imp.values[::-1], color="#4C72B0")
    ax.set_xlabel("Mean decrease in impurity")
    save(fig, out, "fig3_9_feature_importance.png")

    # Fig 3.10 - single-transaction latency distribution
    rf.set_params(n_jobs=1)
    sample = X_te.sample(500, random_state=1)
    lat = []
    for i in range(len(sample)):
        t0 = time.perf_counter()
        rf.predict_proba(sample.iloc[[i]])
        lat.append((time.perf_counter() - t0) * 1000)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(lat, bins=30, color="#4C72B0")
    ax.axvline(np.mean(lat), color="#C44E52", ls="--",
               label=f"mean {np.mean(lat):.2f} ms")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Calls")
    ax.legend()
    save(fig, out, "fig3_10_latency_distribution.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/creditcard.csv")
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    main(a.data, a.out)
