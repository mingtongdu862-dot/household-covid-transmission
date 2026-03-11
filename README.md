# Household SARS-CoV-2 Secondary Transmission Risk Prediction

Source code for the paper:

> **TabPFN for Household SARS-CoV-2 Secondary Transmission Risk Prediction:
> A National Register-Based Study with Interpretability Analysis**

---

## Overview

This repository implements a machine-learning pipeline that predicts whether
a household will experience at least one secondary SARS-CoV-2 transmission
event, given demographic, socio-economic, and clinical features of its members.

The study uses data from the Swedish national health registers covering the
pre-vaccination period (2020), comprising 252,472 households.
The primary model is a custom bagging ensemble of **TabPFN v2.5** classifiers.
A **Random Forest** baseline is also provided for comparison.

A three-level interpretability framework is applied to the trained ensemble:
global permutation importance, risk-stratified subgroup analysis, and local
SHAP waterfall plots.

---

## Repository Structure

```
.
├── household_mapping.py          # Step 1 — build household-to-member mapping
├── feature_engineering.py        # Step 2 — aggregate and encode features
├── tabpfn_config.py              # Shared configuration for TabPFN scripts
├── tabpfn_ensemble.py            # TabPFN bagging ensemble class definition
├── random_forest_baseline.py     # Random Forest baseline (5-fold CV)
└── README.md
```

### Script descriptions

| File | Purpose |
|------|---------|
| `household_mapping.py` | Reads the person-household linkage file and outputs a wide-format CSV mapping each household to its member IDs. |
| `feature_engineering.py` | Aggregates person-level register features to household level, applies chi-square feature selection, and exports stratified 5-fold train/validation/test splits. |
| `tabpfn_config.py` | Centralised hyperparameter file — paths, TabPFN model settings, bagging strategy, class-balance settings, and SHAP configuration. |
| `tabpfn_ensemble.py` | `TabPFNEnsemble` class, `SamplingStrategy` class, and `FeatureSelector` class. Imported by training and explainability scripts. |
| `random_forest_baseline.py` | Trains a Random Forest with configurable imbalance-handling across 5 folds; evaluates on a balanced 50 % positive held-out subset to match the TabPFN evaluation protocol. |

---

## Requirements

### Python version
Python ≥ 3.9

### Dependencies

```
tabpfn>=2.0
torch
scikit-learn
imbalanced-learn
pandas
numpy
tqdm
psutil
shap
```

Install all dependencies with:

```bash
pip install tabpfn torch scikit-learn imbalanced-learn pandas numpy tqdm psutil shap
```

### Hardware
The TabPFN ensemble was trained on an **NVIDIA A100-PCIE-40GB** GPU.
A CUDA-capable GPU with ≥ 16 GB VRAM is recommended; the code falls back to
CPU if no GPU is available (substantially slower).

### TabPFN model weights
Download the TabPFN v2.5 checkpoint and place it at:

```
./tabpfn_weights/tabpfn-v2.5-classifier-v2.5_default.ckpt
```

Or update `MODEL_PATH` in `tabpfn_config.py` to point to your local copy.

---

## Usage

Run the scripts in the following order.

### Step 1 — Household member mapping

```bash
python household_mapping.py
```

**Input:** `Features_Selected_Data/HushallPerson_2019.csv`  
**Output:** `Feature_Tables/household_member.csv`

Reads the person-household linkage table, filters out households outside
the valid size range [1, 200], and writes a wide-format CSV where each row
is a household and each column (`member_1`, `member_2`, …) holds one member ID.

---

### Step 2 — Feature engineering

```bash
python feature_engineering.py
```

**Input:**
- `Feature_Tables/Raw_Feature_Secondary_Case.csv` (person-level features)
- `Feature_Tables/household_member.csv` (output of Step 1)

**Output:** `Encoded_Household_Features_Full/`
- `train_fold_{1..5}.csv`
- `val_fold_{1..5}.csv`
- `test_fold_{1..5}.csv`
- `folds_info.txt`

The pipeline:
1. Aggregates person-level demographic, socio-economic, and clinical count
   features to household level (age, income, housing, TRYGG vulnerability
   scores, prescription and diagnosis counts).
2. Encodes categorical variables (label encoding, frequency encoding,
   one-hot encoding).
3. Applies log1p transforms to skewed continuous features.
4. Selects clinical binary features via chi-square test
   (p < 0.02, max-frequency < 0.95).
5. Standardises continuous features with `StandardScaler` fitted on each
   training fold.
6. Exports stratified 5-fold splits (90 % train+val / 10 % test).

Key configuration parameters (edit `Config` class in `feature_engineering.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TEST_SIZE` | 0.1 | Fraction held out as test set |
| `N_FOLDS` | 5 | Number of cross-validation folds |
| `CHI2_P_VALUE_THRESHOLD` | 0.02 | Chi-square significance threshold |
| `MISSING_RATE_THRESHOLD` | 20 | Max % missing before dropping a column |

---

### Step 3a — TabPFN ensemble training

Edit `tabpfn_config.py` to set paths and hyperparameters, then run your
own training script that imports `TabPFNEnsemble` from `tabpfn_ensemble.py`.

Key configuration sections in `tabpfn_config.py`:

**Bagging strategy (`ENSEMBLE_CONFIG`):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_bags` | 8 | Number of base TabPFN models |
| `bag_sample_size` | 40,000 | Training tokens per bag |
| `bagging_strategy` | `'stratified_random'` | Bag partitioning method |
| `balance_strategy` | `'combined'` | Class-balance method within bags |
| `target_ratio` | 0.6 | Target minority-class fraction per bag |
| `ensemble_method` | `'soft_voting'` | Aggregation method |

**SHAP explainability (`SHAP_CONFIG`):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `strategy` | `'representative'` | SHAP computation strategy |
| `n_background` | 50 | Background samples for KernelExplainer |
| `n_explain_global` | 1,000 | Samples for global importance estimation |

Example training snippet:

```python
from tabpfn_config import TABPFN_PARAMS, ENSEMBLE_CONFIG, TABPFN_MAX_SAMPLES, TABPFN_MAX_FEATURES
from tabpfn_ensemble import TabPFNEnsemble

ensemble = TabPFNEnsemble(
    config=ENSEMBLE_CONFIG,
    tabpfn_params=TABPFN_PARAMS,
    max_samples=TABPFN_MAX_SAMPLES,
    max_features=TABPFN_MAX_FEATURES,
)
ensemble.fit(X_train, y_train, feature_names)
proba = ensemble.predict_proba(X_test, feature_names)
```

---

### Step 3b — Random Forest baseline

```bash
python random_forest_baseline.py
```

**Input:** `Encoded_Household_Features_Full/` (output of Step 2)  
**Output:** `RF_results_balanced/`

Trains a 200-tree Random Forest across all 5 folds with configurable
imbalance handling.  Evaluation is performed on a stratified 50 % positive
subset of 4,000 samples to match the TabPFN evaluation protocol.

Change `IMBALANCE_STRATEGY` at the top of the script to try different
handling approaches (`'class_weight'`, `'smote'`, `'undersample'`, etc.).

---

## Experimental Setup

| Setting | Value |
|---------|-------|
| Dataset period | Pre-vaccination, 2020 |
| Total households | 252,472 |
| Positive class (secondary transmission) | 18.8 % |
| Features | 295 (after selection) |
| Train / test split | 90 % / 10 % |
| Evaluation subset | n = 4,000, 50 % positive |
| TabPFN bags | K = 8 |
| Samples per bag | 40,000 |
| Positive ratio per bag | 60 % |
| GPU | NVIDIA A100-PCIE-40GB |
| Training time (TabPFN ensemble) | ~160 s |

---

## Key Results (test set, 5-fold mean ± SD)

| Model | Recall₊ | F1₊ | MCC | Macro AUC |
|-------|---------|-----|-----|-----------|
| TabPFN Ensemble | 0.905 ± 0.016 | 0.677 ± 0.003 | 0.186 ± 0.003 | 0.626 ± 0.002 |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | 0.627 |

---

## Data Availability

The underlying Swedish register data (SWECOV) are not publicly available
owing to patient privacy regulations.  Code is provided for methodological
transparency; researchers with access to equivalent national register data
may adapt the pipeline accordingly.

---

## License

This code is released for research purposes. See `LICENSE` for details.
