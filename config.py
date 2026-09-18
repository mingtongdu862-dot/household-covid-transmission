"""
config.py
=========
Centralised configuration for the TabPFN ensemble model and XAI analysis.

All paths, hyperparameters, and strategy settings for the TabPFN-based
household secondary transmission prediction pipeline are defined here.
Import this module at the top of ``tabpfn_ensemble.py`` and ``tabpfn_xai.py``
to ensure consistent settings across training and explainability runs.

Sections
--------
PATH_CONFIGURATION
    Input data directory, output directory, and optional model-weight path.
TABPFN_HARDWARE_AND_MODEL_PARAMETERS
    Device selection, number of internal estimators, and version-specific limits.
ENSEMBLE_STRATEGY_CONFIGURATION
    Bagging strategy, bag size, class-balancing strategy and target ratio.
TRAINING_CONFIGURATION
    Prediction batch size.

Kernel SHAP sample sizes and approximation settings for the explainability
pipeline (global/local) live in ``tabpfn_xai.py`` itself
(``GLOBAL_SHAP_CONFIG``, ``LOCAL_SHAP_CONFIG``), since that module is the
only consumer.
"""

import os

# ===========================================================================
# PATH CONFIGURATION
# ===========================================================================
FOLDS_PATH = 'Encoded_Household_Features_Full'
OUTPUT_DIR = 'TabPFN_XAI_Results_Full/'
MODEL_PATH = './tabpfn_weights/tabpfn-v2.5-classifier-v2.5_default.ckpt'

# Data column configuration
DELETED_COLS = []  # Additional columns to delete (if any)
DROP_COLS_BASE = ['household_id', 'IndexDate_household']

# ===========================================================================
# TABPFN HARDWARE AND MODEL PARAMETERS
# ===========================================================================
TABPFN_MAX_SAMPLES = 50000    # TabPFN v2.5 hard limit
TABPFN_MAX_FEATURES = 2000    # TabPFN v2.5 hard limit

TABPFN_PARAMS = {
    'device': 'cuda',
    'n_estimators': 8,
    'ignore_pretraining_limits': False,  # Must be False to enforce v2.5 limits
}

# Require the local checkpoint rather than silently falling back to
# TabPFN's default network download. This pipeline is meant to run in
# network-isolated environments, where a silent fallback here would not
# fail until the first TabPFNClassifier(...) call deep inside training or
# XAI -- potentially after hours of unrelated work -- and could hang
# rather than error out cleanly if outbound network access is blocked
# rather than merely refused. Failing fast at import time makes a missing
# or misplaced checkpoint obvious immediately instead of much later.
if MODEL_PATH is not None:
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"TabPFN checkpoint not found at MODEL_PATH={MODEL_PATH!r} "
            f"(resolved from cwd={os.getcwd()!r}). This pipeline requires "
            f"the local checkpoint -- it does not fall back to downloading "
            f"one, since that would hang or fail in network-isolated "
            f"environments. Place the checkpoint at this path (or update "
            f"MODEL_PATH) before running training or tabpfn_xai.py.")
    TABPFN_PARAMS['model_path'] = MODEL_PATH

# ===========================================================================
# ENSEMBLE STRATEGY CONFIGURATION (Core Hyperparameters)
# ===========================================================================
ENSEMBLE_CONFIG = {
    # --- Bagging Strategy ---
    'bagging_strategy': 'stratified_random',  # Options: 'stratified_random', 'bootstrap', 'diversity'
    'n_bags': 8,                              # Number of base models
    'bag_sample_size': 40000,                 # Samples per bag (leave buffer for prediction)
    'bag_overlap': 0.0,                       # Overlap ratio between bags (0.0=no overlap, 0.5=50% shared)
    
    # --- Feature Strategy (if features > TABPFN_MAX_FEATURES) ---
    'feature_strategy': 'all',                # Options: 'random_groups', 'importance_groups', 'correlation_groups', 'all'
    # 'n_feature_groups': 3,                  # Number of feature groups (if applicable)
    # 'feature_overlap': 0.1,                 # Overlap between feature groups
    
    # --- Ensemble Method ---
    'ensemble_method': 'soft_voting',         # Options: 'soft_voting', 'weighted_voting', 'median'
    'use_oob_weighting': True,                # Use out-of-bag samples to compute weights
    
    # --- Class Balance Strategy ---
    'balance_classes': True,                  # Enable class balancing (IMPORTANT for imbalanced data!)
    'balance_strategy': 'combined',        # Options: 'undersample', 'oversample', 'combined'
    'target_ratio': 0.6,                      # Target ratio for minority class (0.5 = 50/50 balance)
    
    # --- Other ---
    'random_state': 42,
}

# ===========================================================================
# TRAINING CONFIGURATION
# ===========================================================================
PREDICT_BATCH_SIZE = 6000

# ===========================================================================
# ENVIRONMENT CONFIGURATION
# ===========================================================================
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Belt-and-suspenders for network-isolated environments: the MODEL_PATH
# check above already ensures TabPFNClassifier is constructed with a local
# checkpoint, but if the underlying tabpfn/huggingface_hub stack still
# probes the network for anything else (version checks, telemetry, hub
# metadata), these tell it to stay offline instead of hanging on a
# blocked connection. Harmless no-ops if unused. setdefault() so an
# environment variable already set in the shell is never overridden.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
