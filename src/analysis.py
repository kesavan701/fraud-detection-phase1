"""Phase I exploratory analysis and preliminary baseline.

Dataset: ULB credit card fraud dataset (creditcard.csv, 284,807 rows).
Reproduces every number reported in Sections 3.4 and 3.5 of the Phase I report.

Usage:
    python src/analysis.py [--data data/creditcard.csv]
"""

import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

SEED = 42


def main(path: str) -> None:
    df = pd.read_csv(path)
    fraud, legit = df[df.Class == 1], df[df.Class == 0]

    # ---- 1. Data quality -------------------------------------------------
    print("Shape:", df.shape, "| missing values:", int(df.isna().sum().sum()))
    dups = df.duplicated()
    print("Duplicate rows:", int(dups.sum()),
          "| of which fraud:", int(df[dups].Class.sum()))
    print("Zero-amount rows:", int((df.Amount == 0).sum()),
          "| of which fraud:", int((fraud.Amount == 0).sum()))

    # ---- 2. Class balance and amount statistics --------------------------
    print(df.Class.value_counts(), "fraud %:", round(df.Class.mean() * 100, 4))
    print(df.groupby("Class").Amount.describe().round(2))

    # ---- 3. Temporal pattern (relative hour) -----------------------------
    df["Hour"] = (df.Time // 3600) % 24
    hourly = df.groupby("Hour").agg(n=("Class", "size"), frauds=("Class", "sum"))
    hourly["fraud_rate_%"] = (hourly.frauds / hourly.n * 100).round(3)
    print(hourly)

    # ---- 4. Feature separability (Kolmogorov-Smirnov) --------------------
    features = [c for c in df.columns if c not in ("Class", "Hour")]
    ks = pd.Series({f: ks_2samp(fraud[f], legit[f]).statistic for f in features})
    print(ks.sort_values(ascending=False).round(4))
    print(df[features].corrwith(df.Class).sort_values().round(4))

    # ---- 5. Preliminary baseline -----------------------------------------
    X, y = df[features], df.Class
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=SEED)

    threshold = X_tr.Amount[y_tr == 1].median()  # naive rule
    rule_pred = (X_te.Amount > threshold).astype(int)

    rf = RandomForestClassifier(random_state=SEED, n_jobs=-1).fit(X_tr, y_tr)
    proba = rf.predict_proba(X_te)[:, 1]
    rf_pred = (proba >= 0.5).astype(int)

    for name, pred in [("Rule (Amount > %.2f)" % threshold, rule_pred),
                       ("Random Forest", rf_pred)]:
        print(name, "P=%.2f%% R=%.2f%% F1=%.3f flagged=%d" % (
            precision_score(y_te, pred) * 100, recall_score(y_te, pred) * 100,
            f1_score(y_te, pred), pred.sum()),
            "CM(tn,fp,fn,tp)=", confusion_matrix(y_te, pred).ravel())
    print("RF ROC-AUC=%.3f  PR-AUC(AP)=%.3f" % (
        roc_auc_score(y_te, proba), average_precision_score(y_te, proba)))
    print(pd.Series(rf.feature_importances_, index=features)
            .sort_values(ascending=False).head(10).round(4))

    # ---- 6. Single-transaction inference latency -------------------------
    rf.set_params(n_jobs=1)
    sample = X_te.sample(500, random_state=1)
    lat = []
    for i in range(len(sample)):
        t0 = time.perf_counter()
        rf.predict_proba(sample.iloc[[i]])
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    print("Latency ms: mean=%.2f p50=%.2f p95=%.2f p99=%.2f max=%.2f" % (
        lat.mean(), *np.percentile(lat, [50, 95, 99]), lat.max()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/creditcard.csv",
                        help="path to creditcard.csv")
    main(parser.parse_args().data)
