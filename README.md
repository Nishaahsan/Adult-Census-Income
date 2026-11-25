# Adult Census Income — Supervised, Unsupervised & Semi-Supervised Learning

An end-to-end machine learning project demonstrating how **one binary classification dataset** can be used across **three different learning paradigms** — Supervised, Unsupervised, and Semi-Supervised Learning — using the **Adult Census Income** dataset.

## 📌 Project Overview

This project uses the same dataset and the same underlying algorithm family (`RandomForestClassifier`) across three learning strategies to compare how model performance changes as label availability decreases:

| Learning Type | Labels Used | Algorithm |
|---|---|---|
| Supervised | 100% | `RandomForestClassifier` (default params) |
| Semi-Supervised | 10% labeled + 90% unlabeled | `SelfTrainingClassifier` (wraps `RandomForestClassifier`) |
| Unsupervised | 0% (labels used only for post-hoc validation) | `KMeans` |

## 📊 Dataset

**Adult Census Income** ([Kaggle link](https://www.kaggle.com/datasets/uciml/adult-census-income))

- **Size:** ~3.8 MB
- **Rows:** 32,561
- **Features:** 14 (mix of numeric and categorical — age, workclass, education, occupation, hours-per-week, etc.)
- **Target:** `income` — binary (`<=50K`, `>50K`)
- **Why this dataset:** small footprint, clean binary target, realistic mix of categorical/numeric features, and enough structure to be meaningfully explored across all three learning paradigms.

## 🧠 Methodology

### 1. Supervised Learning
- Missing values (`?`) dropped
- Categorical features label-encoded
- 80/20 train-test split (`random_state=42`, stratified)
- `RandomForestClassifier()` trained with **default hyperparameters** — no tuning
- Evaluated on: **EM, ACC, MCC, AUROC**

### 2. Unsupervised Learning
- The `income` column is **fully removed** before training (verified explicitly in code)
- Features scaled with `StandardScaler`
- `KMeans(n_clusters=2)` clustering trained on features only — no labels involved
- Evaluated on:
  - **Silhouette Score** (internal, label-free metric)
  - **Adjusted Rand Index** (external validation only — labels used *after* training to check alignment with true classes, never during training)
- PCA used for 2D visualization of clusters vs. true labels

### 3. Semi-Supervised Learning
- Starting from the same 80% training split, **90% of labels are masked** (`-1`) to simulate scarce labeled data
- `SelfTrainingClassifier` wraps the same `RandomForestClassifier`, iteratively pseudo-labeling high-confidence unlabeled samples
- Evaluated on the same held-out test set using: **EM, ACC, MCC, AUROC**

## 📈 Results Summary

| Metric | Supervised (100% labels) | Semi-Supervised (10% labels) |
|---|---|---|
| EM (Exact Match) | ~0.853 | ~0.829 |
| ACC (Accuracy) | ~0.853 | ~0.829 |
| MCC | ~0.610 | ~0.543 |
| AUROC | ~0.907 | ~0.873 |

| Unsupervised Metric | Score |
|---|---|
| Silhouette Score | ~0.172 |
| Adjusted Rand Index | ~0.038 |

**Key finding:** the semi-supervised model retains most of the fully-supervised model's predictive power using only **10% of the labels** — demonstrating real value when labeling data is expensive. Unsupervised clustering finds structure in the feature space, but that structure only weakly aligns with the income boundary (low ARI) — an expected and honest result, since KMeans has no knowledge of the label it's being compared against.

## ⚠️ Why Unsupervised Learning Doesn't Use EM/ACC/MCC/AUROC

These four metrics require a predicted class label (or probability) to compare against a true label. KMeans clusters have no inherent class meaning — "Cluster 0" isn't tied to `<=50K` or `>50K` in any consistent way, and there's no `predict_proba()` output for AUROC. That's why clustering quality is measured with **Silhouette Score** and **Adjusted Rand Index** instead — this is standard practice, not a limitation of the code.

## 🛠️ Tech Stack

- Python 3
- pandas, numpy
- scikit-learn (`RandomForestClassifier`, `KMeans`, `SelfTrainingClassifier`, `StandardScaler`, `PCA`, `LabelEncoder`)
- matplotlib, seaborn

## 📁 Repository Structure

```
adult-census-income-ml/
│
├── 📓 notebook.ipynb              # Full Kaggle notebook (all 15 code blocks)
│
├── 📄 README.md                   # Project documentation (this file)
│
├── 📋 requirements.txt            # Python dependencies
│
├── 📂 data/
│   └── adult.csv                  # Dataset (or link if not committed directly)
│
├── 📂 images/
│   ├── confusion_matrix.png       # Supervised learning confusion matrix
│   ├── roc_curve.png              # Supervised learning ROC curve
│   ├── kmeans_clusters.png        # Unsupervised clustering (PCA-reduced)
│   └── metrics_comparison.png     # Supervised vs Semi-Supervised comparison chart
│
└── 📂 results/
    └── summary_metrics.csv        # Final EM, ACC, MCC, AUROC scores exported
```
