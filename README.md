# Household SARS-CoV-2 Secondary Transmission Risk Prediction

Source code for the paper:

> **Household SARS-CoV-2 Secondary Transmission Risk Prediction Using TabPFN with Explainability on Swedish National Register Data**

## Overview

This repository contains the full code for a study that predicts whether secondary COVID-19 transmission will occur within a household, using data from Swedish national health and socioeconomic registers during the first wave of 2020 (pre-vaccination period). The pipeline covers:

1. **Data preprocessing** – time-filtering, feature selection, and population-index construction from raw register CSVs.
2. **Household mapping** – building a wide-format household-to-member table.
3. **Feature extraction** – individual-level static (demographic/socioeconomic) and dynamic (medical-history) features.
4. **Feature aggregation** – aggregating person-level features to the household level, encoding, and producing stratified k-fold splits.
5. **Model training** – TabPFN ensemble (primary model) plus Logistic Regression, Random Forest, and XGBoost baselines.
6. **Explainability analysis** – global permutation importance, Kernel SHAP (subgroup and local waterfall plots).

---

## Repository Structure

```
.
├── config.py                          # Centralised configuration for TabPFN and XAI
│
├── data_preprocessing.py              # Step 1 – Time-filtering, feature selection,
│                                      #          CSV→dict conversion, population index
├── household_mapping.py               # Step 2 – Household-to-member mapping table
├── feature_extraction.py              # Step 3 – Individual-level feature extraction
├── feature_aggregation.py             # Step 4 – Household-level aggregation, encoding,
│                                      #          chi-square selection, k-fold splits
│
├── tabpfn_ensemble.py                 # Step 5a – TabPFN ensemble class (SamplingStrategy,
│                                      #           FeatureSelector, TabPFNEnsemble)
├── tabpfn_train.py                    # Step 5b – TabPFN 5-fold CV training & evaluation
│                                      #           (equivalent to baseline scripts)
├── tabpfn_xai.py                      # Step 5c – Explainability analysis (PI + SHAP)
│
├── baseline_logistic_regression.py    # Logistic Regression baseline
├── baseline_random_forest.py          # Random Forest baseline
└── baseline_xgboost.py               # XGBoost baseline
```

---

## Pipeline Execution Order

Run the scripts in the following order to reproduce the study from raw register data:

```bash
# Step 1: Preprocess raw register data
python data_preprocessing.py

# Step 2: Build household membership table
python household_mapping.py

# Step 3: Extract individual-level features
python feature_extraction.py

# Step 4: Aggregate to household level and create k-fold splits
python feature_aggregation.py

# Step 5a: Train and evaluate the TabPFN ensemble (5-fold CV, matches baseline format)
python tabpfn_train.py

# Step 5b: Run explainability analysis on the trained ensemble
python tabpfn_xai.py

# Note: tabpfn_ensemble.py is the class library imported by both scripts above.

# Optional: Train baseline models
python baseline_logistic_regression.py
python baseline_random_forest.py
python baseline_xgboost.py
```

---

## Data Requirements

This pipeline was designed for **Swedish national register data** accessed through SWECOV. The expected input files are:

| File | Contents |
|------|----------|
| `FHM_SMINET.csv` | Laboratory-confirmed COVID-19 diagnoses (diagnosis dates) |
| `Population_PersonNr_20221231.csv` | Population register (sex, birth year/month) |
| `Fodelseuppg_20201231.csv` | Country of birth and immigration background |
| `HushallPerson_2019.csv` | Household membership (household ID per person) |
| `HushallBoende_2019.csv` | Housing type and living area per person |
| `Individ_2019.csv` | Disposable personal and family income |
| `Inera_VPTU_Coronadata.csv` | Primary-care contact reasons (2019–2020) |
| `SWECOV_SOS_LMED.csv` | Dispensed medications – ATC codes (2019–2020) |
| `SWECOV_SOS_OV.csv` | Outpatient diagnoses – ICD-10 (all years) |
| `SWECOV_SOS_SV.csv` | Inpatient diagnoses – ICD-10 (all years) |
| `SWECOV_SOS_DORS.csv` | Death dates (2020) |
| `SWECOV_SOS_SOL.csv` | Elderly care (TRYGG indicators, 2018–2020) |
| `RTB2019.csv` | 2019 population register for 2020 base population |

Set the path to the raw data directory in `data_preprocessing.py`:

```python
class Config:
    INPUT_DIR_RAW = '/path/to/raw_register_data'
```

> **Note:** The register data are not redistributable. Access requires a formal data application to the Swedish authority holding the relevant registers.

---

## Configuration

All key settings are centralised in `config.py`. The most important parameters are:

```python
# Input / output
FOLDS_PATH  = 'Encoded_Household_Features_Full'  # K-fold CSV directory
OUTPUT_DIR  = 'TabPFN_XAI_Results_Full/'          # Results and plots

# TabPFN limits (v2.5 large-samples checkpoint)
TABPFN_MAX_SAMPLES  = 50_000
TABPFN_MAX_FEATURES = 2_000

# Ensemble strategy
ENSEMBLE_CONFIG = {
    'n_bags'          : 8,       # Number of bags
    'bag_sample_size' : 40_000,  # Training samples per bag
    'balance_classes' : True,
    'target_ratio'    : 0.6,     # 60 % positive per bag (oversampling)
    ...
}

# SHAP analysis
SHAP_CONFIG = {
    'n_background'      : 50,
    'n_explain_global'  : 1_000,
    'max_evals'         : 120,
    ...
}
```


---

## Model Architecture

### TabPFN Ensemble (Primary Model)

`tabpfn_ensemble.py` trains K = 8 TabPFN v2.5 (`large-samples` checkpoint) classifiers in a bagging framework.

Each bag draws 40,000 training samples from the full training set, resampled to a 60 % positive ratio (24,000 positive + 16,000 negative). This oversampling strategy is the key design choice that enables Recall₊ = 0.905 on the balanced test subset.

Inference uses soft-voting (arithmetic mean of predicted probabilities) across all bags, optionally weighted by each bag's OOB AUC score.

**Requires an NVIDIA GPU.** Training 8 bags takes ≈ 160 seconds on an A100-40 GB.

### Baselines

All three baseline scripts (Logistic Regression, Random Forest, XGBoost) follow the same evaluation protocol:

1. Train on full imbalanced training folds with `class_weight='balanced'` (or `scale_pos_weight` for XGBoost).
2. Evaluate on a balanced held-out subset (n = 4,000, 50 % positive) to ensure fair comparison with TabPFN.

---

## Explainability

`tabpfn_xai.py` implements a three-level interpretability framework:

| Level | Method | Purpose |
|-------|--------|---------|
| Global | Permutation Importance (AUC-drop, n = 3,000, R = 5) | Which features matter overall? |
| Subgroup | PI + Kernel SHAP on risk strata | Do importance patterns differ by predicted risk? |
| Local | Kernel SHAP waterfall plots | Why did the model assign this score to this household? |

The risk strata are defined by the ensemble's predicted probability:
- **High risk:** ŷ ≥ 0.7
- **Moderate risk:** 0.3 ≤ ŷ < 0.7
- **Low risk:** ŷ < 0.3

---

## Results Summary

| Metric | Logistic Reg. | Random Forest | XGBoost | **TabPFN** |
|--------|:---:|:---:|:---:|:---:|
| Macro AUC | 0.620 ± 0.003 | 0.612 ± 0.002 | **0.627** ± 0.003 | 0.626 ± 0.002 |
| MCC | 0.170 ± 0.008 | 0.134 ± 0.006 | 0.176 ± 0.011 | **0.186** ± 0.003 |
| Recall₊ | 0.603 ± 0.014 | 0.420 ± 0.006 | 0.552 ± 0.007 | **0.905** ± 0.016 |
| F1₊ | 0.592 ± 0.008 | 0.491 ± 0.005 | 0.572 ± 0.006 | **0.677** ± 0.003 |

Evaluation on a balanced held-out subset (n = 4,000; 50 % positive prevalence).

---

## Requirements

```
python >= 3.9
tabpfn >= 2.5          # TabPFN v2.5 with large-samples checkpoint
torch                  # PyTorch (GPU recommended)
scikit-learn
xgboost
imbalanced-learn
shap
pandas
numpy
tqdm
psutil
```

Install dependencies:

```bash
pip install tabpfn torch scikit-learn xgboost imbalanced-learn shap pandas numpy tqdm psutil
```

The TabPFN v2.5 `large-samples` checkpoint can be specified via the `MODEL_PATH` variable in `config.py`:

```python
MODEL_PATH = './tabpfn_weights/tabpfn-v2.5-classifier-v2.5_default.ckpt'
```

---

## Expected Directory Layout After Full Pipeline

```
.
├── Time_Filtered_Data/              # Step 1 output: time-filtered CSVs
├── Features_Selected_Data/          # Step 1 output: selected-feature CSVs + pickle dicts
├── Index(InD=14,ExD=2)/             # Step 1 output: population index pickles
├── PopIndex/                        # Step 1 output: infected/healthy classification
├── Feature_Tables/                  # Steps 2–3 output: household map + raw feature CSV
├── Household_Features/              # Step 4 intermediate: unaggregated household table
├── Encoded_Household_Features_Full/ # Step 4 output: k-fold train/val/test CSVs
├── TabPFN_XAI_Results_Full/         # Steps 5a–5b output: metrics, SHAP plots, PI charts
├── LR_results_balanced_Full_2/      # Logistic Regression results
├── RF_results_balanced_Full_2/      # Random Forest results
└── XGB_results_balanced_Full_2/     # XGBoost results
```
