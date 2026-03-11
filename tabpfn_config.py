"""
TabPFN Ensemble — Unified Configuration

Centralised hyperparameter and path configuration for the TabPFN ensemble
training and explainability pipeline.  All settings consumed by
tabpfn_ensemble.py and any downstream training scripts should be imported
from this module.

Usage
-----
    from tabpfn_config import (
        FOLDS_PATH, OUTPUT_DIR, TABPFN_PARAMS,
        ENSEMBLE_CONFIG, SHAP_CONFIG,
    )
"""

import os


# ===========================================================================
# PATH CONFIGURATION
# ===========================================================================

FOLDS_PATH  = 'Encoded_Household_Features_Full'   # Directory of K-fold CSV splits
OUTPUT_DIR  = 'TabPFN_XAI_Results_Full/'           # Directory for model outputs and results
MODEL_PATH  = './tabpfn_weights/tabpfn-v2.5-classifier-v2.5_default.ckpt'

# Columns to drop before training (identifiers and date fields)
DELETED_COLS   = []   # Additional columns to exclude, if any
DROP_COLS_BASE = ['household_id', 'IndexDate_household']


# ===========================================================================
# TABPFN HARDWARE AND MODEL PARAMETERS
# ===========================================================================

TABPFN_MAX_SAMPLES  = 50_000   # TabPFN v2.5 hard upper limit on training tokens
TABPFN_MAX_FEATURES = 2_000    # TabPFN v2.5 hard upper limit on features

TABPFN_PARAMS = {
    'device':                    'cuda',
    'n_estimators':              8,
    'ignore_pretraining_limits': False,  # Must remain False to enforce v2.5 limits
}

# Attach custom model weights if the checkpoint file exists
if MODEL_PATH is not None and os.path.isfile(MODEL_PATH):
    TABPFN_PARAMS['model_path'] = MODEL_PATH


# ===========================================================================
# ENSEMBLE STRATEGY CONFIGURATION
# ===========================================================================

ENSEMBLE_CONFIG = {
    # ── Bagging ───────────────────────────────────────────────────────────────
    # Strategy used to partition training data across bags.
    # Options: 'stratified_random' | 'bootstrap' | 'diversity'
    'bagging_strategy': 'stratified_random',

    # Number of base TabPFN models in the ensemble.
    'n_bags': 8,

    # Number of training samples per bag.
    # Should remain below TABPFN_MAX_SAMPLES minus any prediction overhead.
    'bag_sample_size': 40_000,

    # Fraction of training data shared between bags.
    # 0.0 = non-overlapping (default); 0.5 = 50 % overlap.
    'bag_overlap': 0.0,

    # ── Feature strategy (when n_features > TABPFN_MAX_FEATURES) ─────────────
    # Options: 'random_groups' | 'importance_groups' | 'correlation_groups' | 'all'
    'feature_strategy': 'all',
    # 'n_feature_groups': 3,      # Number of feature groups (if applicable)
    # 'feature_overlap':  0.1,    # Overlap between feature groups

    # ── Ensemble aggregation ──────────────────────────────────────────────────
    # Options: 'soft_voting' | 'weighted_voting' | 'median'
    'ensemble_method':   'soft_voting',

    # Weight each base model by its out-of-bag (OOB) AUC score.
    'use_oob_weighting': True,

    # ── Class balancing ───────────────────────────────────────────────────────
    # Enable oversampling / undersampling within each bag to counteract
    # the natural class imbalance (~18.8 % positive households).
    'balance_classes':  True,

    # Options: 'undersample' | 'oversample' | 'combined'
    'balance_strategy': 'combined',

    # Target minority-class fraction within each bag (0.6 → 60 % positive).
    'target_ratio': 0.6,

    # ── Reproducibility ───────────────────────────────────────────────────────
    'random_state': 42,
}


# ===========================================================================
# TRAINING CONFIGURATION
# ===========================================================================

# Batch size used during prediction (to avoid GPU OOM errors).
PREDICT_BATCH_SIZE = 5_000

# Fraction of the training set reserved for the held-out test set.
TEST_SAMPLE_RATIO  = 0.1


# ===========================================================================
# SHAP EXPLAINABILITY CONFIGURATION
# ===========================================================================

SHAP_CONFIG = {
    # ── Strategy ──────────────────────────────────────────────────────────────
    # 'representative' uses only the best 1–2 base models for efficiency.
    # Options: 'representative' | 'ensemble_blackbox' | 'aggregated'
    'strategy': 'representative',

    # Number of base models used for SHAP computation.
    'n_representative_models': 1,

    # Criterion for selecting the representative model(s).
    # Options: 'oob_score' | 'random' | index list, e.g. [0, 5, 10]
    'selection_criterion': 'oob_score',

    # ── Sampling ──────────────────────────────────────────────────────────────
    # Number of background samples for the SHAP KernelExplainer.
    'n_background': 50,

    # Number of samples for global SHAP importance estimation.
    'n_explain_global': 1_000,

    # Number of samples for local (per-instance) SHAP waterfall plots.
    'n_explain_local': 5,

    # ── Approximation ────────────────────────────────────────────────────────
    # Maximum number of model evaluations per SHAP explanation.
    'max_evals': 120,

    # Mini-batch size passed to the SHAP explainer.
    'batch_size': 10,

    # ── Quality / speed trade-off ─────────────────────────────────────────────
    'use_fast_shap': True,   # Use SHAP approximations rather than exact values
    'cache_shap':    True,   # Cache computed SHAP values to disk
}


# ===========================================================================
# ENVIRONMENT CONFIGURATION
# ===========================================================================

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
