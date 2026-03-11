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
    Prediction batch size and test-set sampling ratio.
SHAP_ANALYSIS_CONFIGURATION
    Strategy, sample sizes, and approximation settings for Kernel SHAP.
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

# If custom model weight path exists
if MODEL_PATH is not None and os.path.isfile(MODEL_PATH):
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
PREDICT_BATCH_SIZE = 5000
TEST_SAMPLE_RATIO = 0.1  # Test set sampling ratio (relative to training set size)

# ===========================================================================
# SHAP ANALYSIS CONFIGURATION (for xAI)
# ===========================================================================
SHAP_CONFIG = {
    # Strategy selection
    'strategy': 'representative',  # Options: 'representative', 'ensemble_blackbox', 'aggregated'
    
    # Representative model strategy (FASTEST)
    'n_representative_models': 1,  # Use only 1-2 best models
    'selection_criterion': 'oob_score',  # 'oob_score', 'random', or index list [0,5,10]
    
    # Sampling (CRITICAL for speed)
    'n_background': 50,        # Background samples (50-100 is enough)
    'n_explain_global': 1000,  # Global SHAP samples (1000-2000)
    'n_explain_local': 5,    # Local SHAP samples (100-500)
    
    # Approximation (CRITICAL for speed)
    'max_evals': 120,          # Kernel evaluations (500-1000, vs 2*n_features)
    'batch_size': 10,         # SHAP batch size
    
    # Quality vs Speed tradeoff
    'use_fast_shap': True,     # Use approximations
    'cache_shap': True,        # Cache SHAP values
}

# ===========================================================================
# ENVIRONMENT CONFIGURATION
# ===========================================================================
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'