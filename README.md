# AI-Driven Full-Stack Platform for Real-Time Financial Fraud Detection and Anomaly Analysis — Phase I

M.E. Computer Science and Engineering project (Phase I), Anna University, Chennai Institute of Technology.

- **Student:** Kesavan M (Reg. No. 210425405005)
- **Supervisor:** Mr. K. Madhavan, M.E., Assistant Professor, CSE
- **Report:** November 2026

## About

The project proposes a platform that scores card transactions for fraud probability in real time using supervised machine learning, and presents the results on an analyst dashboard with a review workflow.

This repository holds the **Phase I code**: the exploratory data analysis (EDA) and the preliminary baseline investigation reported in Chapter 3 (Sections 3.4 and 3.5) of the Phase I report. The full application (REST API, React dashboard, PostgreSQL, Docker) will be added in Phase II.

## Phase I results (held-out test set, 71,202 transactions, 123 frauds)

| Approach | Precision | Recall | F1 | Flagged |
| --- | --- | --- | --- | --- |
| Rule-based (Amount > 9.21) | 0.13% | 50.41% | 0.003 | 48,700 |
| Random Forest (default, untuned) | 94.29% | 80.49% | 0.868 | 105 |

Random Forest: ROC-AUC 0.946, PR-AUC (average precision) 0.841. Single-transaction inference takes about 5.9 ms on one CPU core (max 8.75 ms), well inside the 500 ms budget.

## Dataset

[ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): 284,807 European card transactions from September 2013, 492 of them fraudulent (0.17%). Features `V1`–`V28` are PCA components; `Time` and `Amount` are original.

The file is not stored in this repository. Download `creditcard.csv` from Kaggle and place it in `data/`.

## Setup

```bash
git clone https://github.com/kesavan701/fraud-detection-phase1.git
cd fraud-detection-phase1
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Prints every number reported in Sections 3.4 and 3.5
python src/analysis.py --data data/creditcard.csv

# Regenerates Figures 3.1 - 3.10 into figures/
python src/make_figures.py --data data/creditcard.csv --out figures
```

To save the console output:

```bash
python src/analysis.py > results/analysis_output.txt
```

## Reproducibility

- Train/test split: 75% / 25%, stratified on `Class`, `random_state=42`
- Random Forest: scikit-learn defaults (100 trees), `random_state=42`, threshold 0.5
- Rule baseline: flag if `Amount` > median Amount of training-set frauds
- Latency: 500 test transactions (`random_state=1`), scored one at a time with `n_jobs=1`. Latency varies with hardware; classification results are deterministic.

## Known limitations (addressed in Phase II)

- Single random split instead of cross-validation (only 123 test frauds)
- 1,081 duplicate rows were not removed, so some train/test leakage is possible
- Fixed 0.5 threshold and default hyperparameters
- No class weighting or feature engineering

## Repository structure

```
.
├── src/
│   ├── analysis.py        # EDA + baseline (Report Appendix A)
│   └── make_figures.py    # Figures 3.1 - 3.10
├── data/                  # place creditcard.csv here (git-ignored)
├── figures/               # generated plots
├── results/               # saved console output
├── requirements.txt
└── LICENSE
```

## Roadmap (Phase II)

Cross-validated model tuning, class-imbalance handling, SHAP explanations, Flask REST API, React analyst dashboard, PostgreSQL storage, Docker deployment, and full testing.

## License

MIT — see [LICENSE](LICENSE).
