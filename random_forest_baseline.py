"""
Random Forest Baseline Classifier

Trains a Random Forest classifier on the household-level feature folds
produced by feature_engineering.py and evaluates it on a balanced held-out
subset (50 % positive class) to match the evaluation protocol used for
the TabPFN ensemble.

Imbalance handling
------------------
Six strategies are available via the IMBALANCE_STRATEGY constant:
    'class_weight'      — pass class_weight='balanced' to scikit-learn
    'smote'             — SMOTE oversampling
    'borderline_smote'  — Borderline-SMOTE oversampling
    'adasyn'            — ADASYN adaptive oversampling
    'undersample'       — random undersampling of the majority class
    'smote_tomek'       — SMOTE + Tomek-Links cleaning
    'smote_enn'         — SMOTE + Edited Nearest Neighbours cleaning
    'none'              — no resampling

Output
------
RF_results_balanced/
    household_rf_summary.csv          — per-fold metrics
    household_rf_top50_features.csv   — mean feature importances
    household_rf_full_results.json    — full nested results
    rf_model_fold_{k}.joblib          — saved model checkpoints
    rf_features_fold_{k}.json         — feature name lists
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

warnings.filterwarnings('ignore')


# ===========================================================================
# CONFIGURATION
# ===========================================================================

OUTPUT_DIR = 'RF_results_balanced/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

DELETED_COLS = []   # Additional feature columns to exclude before training

# Imbalance handling strategy applied to the training set.
# See module docstring for available options.
IMBALANCE_STRATEGY = 'class_weight'

# Target minority-class ratio for sampling-based strategies (None = auto).
SAMPLING_RATIO = 0.8

# Evaluation subset configuration — matches the TabPFN evaluation protocol.
EVAL_CONFIG = {
    'eval_size':      4_000,   # Total samples in each evaluation subset
    'eval_pos_ratio': 0.5,     # Fraction of positive samples in each subset
    'random_state':   42,
}

# Random Forest hyperparameters.
RF_PARAMS = {
    'n_estimators':      200,
    'max_depth':         15,
    'min_samples_split': 10,
    'min_samples_leaf':  4,
    'max_features':      'sqrt',
    'random_state':      42,
    'n_jobs':            -1,
    'class_weight':      None,   # Set to 'balanced' when IMBALANCE_STRATEGY='class_weight'
    'bootstrap':         True,
    'oob_score':         True,
}


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_and_preprocess(fold: int):
    """
    Load and preprocess the train / validation / test CSV files for one fold.

    Missing-value imputation is NOT performed here; it is applied later via
    SimpleImputer so that the imputer is fitted only on training data.

    Args:
        fold: Fold index (1-based integer).

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test,
                  feature_names) where all arrays are pandas Series /
        DataFrames.
    """
    base_path = 'Encoded_Household_Features_Full'
    drop_cols = ['household_id', 'IndexDate_household'] + DELETED_COLS
    label_col = 'label'

    train_df = pd.read_csv(f'{base_path}/train_fold_{fold}.csv', encoding='latin1')
    val_df   = pd.read_csv(f'{base_path}/val_fold_{fold}.csv',   encoding='latin1')
    test_df  = pd.read_csv(f'{base_path}/test_fold_{fold}.csv',  encoding='latin1')

    for df in [train_df, val_df, test_df]:
        df.drop(columns=drop_cols, errors='ignore', inplace=True)

    # Derive binary label from secondary-case count
    for df in [train_df, val_df, test_df]:
        df['label'] = (df['secondary_cases_count'] > 0).astype(int)
        df.drop('secondary_cases_count', axis=1, inplace=True)

    y_train = train_df[label_col]
    X_train = train_df.drop(label_col, axis=1)
    y_val   = val_df[label_col]
    X_val   = val_df.drop(label_col, axis=1)
    y_test  = test_df[label_col]
    X_test  = test_df.drop(label_col, axis=1)

    feature_names = X_train.columns.tolist()
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


# ===========================================================================
# EVALUATION UTILITIES
# ===========================================================================

def sample_balanced_eval(
    X: np.ndarray, y: np.ndarray,
    eval_size: int, pos_ratio: float,
    random_state: int, tag: str = '',
) -> tuple:
    """
    Draw a stratified balanced subset from an evaluation array.

    Args:
        X:            Feature array (already imputed).
        y:            Binary label array.
        eval_size:    Total number of samples to draw.
        pos_ratio:    Target fraction of positive samples (e.g. 0.5).
        random_state: NumPy random seed.
        tag:          Label used in log messages.

    Returns:
        Tuple of (X_sampled, y_sampled) ndarrays.
    """
    np.random.seed(random_state)

    n_pos_target = int(eval_size * pos_ratio)
    n_neg_target = eval_size - n_pos_target

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) < n_pos_target:
        print(f'  WARNING [{tag}]: requested {n_pos_target} positives but '
              f'only {len(pos_idx)} available')
        n_pos_target = len(pos_idx)
        n_neg_target = eval_size - n_pos_target

    if len(neg_idx) < n_neg_target:
        print(f'  WARNING [{tag}]: requested {n_neg_target} negatives but '
              f'only {len(neg_idx)} available')
        n_neg_target = len(neg_idx)
        n_pos_target = eval_size - n_neg_target

    chosen_pos = np.random.choice(pos_idx, n_pos_target, replace=False)
    chosen_neg = np.random.choice(neg_idx, n_neg_target, replace=False)
    chosen     = np.concatenate([chosen_pos, chosen_neg])
    np.random.shuffle(chosen)

    actual_pos = (y[chosen] == 1).mean()
    print(f'  [{tag}] sampled {len(chosen)} samples  pos={actual_pos:.1%}')
    return X[chosen], y[chosen]


def apply_imbalance_handling(
    X_train: np.ndarray, y_train: np.ndarray,
    strategy: str = 'smote',
    sampling_ratio=None,
) -> tuple:
    """
    Apply an imbalance-handling strategy to the training data.

    Args:
        X_train:        Training feature array.
        y_train:        Training label array.
        strategy:       Name of the imbalance-handling strategy to apply.
        sampling_ratio: Target ratio for sampling-based strategies.
                        Pass None to use the default ('auto' in imbalanced-learn).

    Returns:
        Tuple of (X_resampled, y_resampled) ndarrays.
    """
    print(f'  Class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}')

    if strategy in ('none', 'class_weight'):
        print(f'  Strategy: {strategy} — no resampling applied')
        return X_train, y_train

    sampling_strategy = 'auto' if sampling_ratio is None else sampling_ratio

    strategy_map = {
        'smote':            SMOTE(sampling_strategy=sampling_strategy,        random_state=42),
        'borderline_smote': BorderlineSMOTE(sampling_strategy=sampling_strategy, random_state=42),
        'adasyn':           ADASYN(sampling_strategy=sampling_strategy,       random_state=42),
        'undersample':      RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42),
        'smote_tomek':      SMOTETomek(sampling_strategy=sampling_strategy,   random_state=42),
        'smote_enn':        SMOTEENN(sampling_strategy=sampling_strategy,     random_state=42),
    }

    if strategy not in strategy_map:
        raise ValueError(f'Unknown imbalance strategy: {strategy!r}')

    print(f'  Applying {strategy} (ratio={sampling_strategy})')
    sampler                = strategy_map[strategy]
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    print(f'  Resampled distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}')
    return X_resampled, y_resampled


def compute_detailed_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
    num_classes: int = 2,
) -> dict:
    """
    Compute a comprehensive suite of binary classification metrics.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Predicted class labels.
        y_prob:      Predicted probability of the positive class.
        num_classes: Number of classes (default 2).

    Returns:
        dict: Macro AUC, PR-AUC, macro/weighted F1, log loss, balanced
              accuracy, Cohen's kappa, MCC, confusion matrix, full
              per-class precision / recall / F1, and per-class AUC / PR-AUC.
    """
    report       = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    auc_macro    = roc_auc_score(y_true, y_prob)
    pr_auc_macro = average_precision_score(y_true, y_prob)
    cm           = confusion_matrix(y_true, y_pred)
    logloss      = log_loss(y_true, np.column_stack([1 - y_prob, y_prob]))
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    kappa        = cohen_kappa_score(y_true, y_pred)
    mcc          = matthews_corrcoef(y_true, y_pred)

    result = {
        'macro_auc':              float(auc_macro),
        'macro_pr_auc':           float(pr_auc_macro),
        'macro_f1':               float(report['macro avg']['f1-score']),
        'weighted_f1':            float(report['weighted avg']['f1-score']),
        'log_loss':               float(logloss),
        'balanced_accuracy':      float(balanced_acc),
        'cohen_kappa':            float(kappa),
        'mcc':                    float(mcc),
        'confusion_matrix':       cm.tolist(),
        'classification_report':  report,
    }

    for cls in range(num_classes):
        result[f'class_{cls}_precision'] = float(report.get(str(cls), {}).get('precision', np.nan))
        result[f'class_{cls}_recall']    = float(report.get(str(cls), {}).get('recall',    np.nan))
        result[f'class_{cls}_f1']        = float(report.get(str(cls), {}).get('f1-score',  np.nan))
        result[f'class_{cls}_support']   = int  (report.get(str(cls), {}).get('support',   0))
        result[f'class_{cls}_auc']       = (
            roc_auc_score(y_true == cls, y_prob) if cls == 1 else np.nan
        )
        result[f'class_{cls}_pr_auc']    = (
            average_precision_score(y_true == cls, y_prob) if cls == 1 else np.nan
        )

    return result


# ===========================================================================
# MAIN TRAINING LOOP
# ===========================================================================

print('\n' + '=' * 80)
print(f'Random Forest Baseline — Imbalance strategy: {IMBALANCE_STRATEGY.upper()}')
print(f'Evaluation: balanced subset | '
      f'size={EVAL_CONFIG["eval_size"]} | '
      f'pos_ratio={EVAL_CONFIG["eval_pos_ratio"]:.0%}')
print('=' * 80)

all_fold_results        = []
feature_importance_list = []

for fold in range(1, 6):
    print(f"\n{'='*60}\nFOLD {fold}\n{'='*60}")

    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     feature_names) = load_and_preprocess(fold)

    # Fit imputer on training data; apply to all splits
    imputer      = SimpleImputer(strategy='mean')
    X_train_imp  = imputer.fit_transform(X_train)
    X_val_imp    = imputer.transform(X_val)
    X_test_imp   = imputer.transform(X_test)

    # Apply training-set imbalance handling
    X_train_bal, y_train_bal = apply_imbalance_handling(
        X_train_imp, y_train,
        strategy=IMBALANCE_STRATEGY,
        sampling_ratio=(SAMPLING_RATIO
                        if IMBALANCE_STRATEGY not in ('none', 'class_weight')
                        else None),
    )

    # Draw balanced evaluation subsets
    print('\n  Sampling balanced evaluation subsets:')
    X_val_bal,  y_val_bal  = sample_balanced_eval(
        X_val_imp, y_val.values,
        eval_size=EVAL_CONFIG['eval_size'],
        pos_ratio=EVAL_CONFIG['eval_pos_ratio'],
        random_state=EVAL_CONFIG['random_state'],
        tag='Val',
    )
    X_test_bal, y_test_bal = sample_balanced_eval(
        X_test_imp, y_test.values,
        eval_size=EVAL_CONFIG['eval_size'],
        pos_ratio=EVAL_CONFIG['eval_pos_ratio'],
        random_state=EVAL_CONFIG['random_state'],
        tag='Test',
    )

    # Set class_weight parameter if applicable
    current_params = RF_PARAMS.copy()
    if IMBALANCE_STRATEGY == 'class_weight':
        current_params['class_weight'] = 'balanced'
        print("  class_weight set to 'balanced'")

    # Train model
    print('Training Random Forest...')
    model = RandomForestClassifier(**current_params)
    model.fit(X_train_bal, y_train_bal)

    if hasattr(model, 'oob_score_'):
        print(f'  OOB Score: {model.oob_score_:.4f}')

    # Evaluate on balanced subsets
    val_prob    = model.predict_proba(X_val_bal)[:, 1]
    val_pred    = model.predict(X_val_bal)
    val_metrics = compute_detailed_metrics(y_val_bal, val_pred, val_prob)

    test_prob    = model.predict_proba(X_test_bal)[:, 1]
    test_pred    = model.predict(X_test_bal)
    test_metrics = compute_detailed_metrics(y_test_bal, test_pred, test_prob)

    # Print per-fold results
    for split_name, metrics in [('VALIDATION', val_metrics), ('TEST', test_metrics)]:
        eval_ratio = EVAL_CONFIG['eval_pos_ratio']
        print(f"\n{'='*60}")
        print(f'FOLD {fold} — {split_name} (balanced {eval_ratio:.0%} positive)')
        print(f"{'='*60}")
        print(f'Macro AUC:        {metrics["macro_auc"]:.4f}')
        print(f'Macro F1:         {metrics["macro_f1"]:.4f}')
        print(f'Weighted F1:      {metrics["weighted_f1"]:.4f}')
        print(f'Log Loss:         {metrics["log_loss"]:.4f}')
        print(f'Balanced Acc:     {metrics["balanced_accuracy"]:.4f}')
        print(f'Cohen Kappa:      {metrics["cohen_kappa"]:.4f}')
        print(f'MCC:              {metrics["mcc"]:.4f}')
        print(f'\nClass 1 (secondary transmission) metrics:')
        print(f'  AUC:            {metrics["class_1_auc"]:.4f}')
        print(f'  Precision:      {metrics["class_1_precision"]:.4f}')
        print(f'  Recall:         {metrics["class_1_recall"]:.4f}')
        print(f'  F1:             {metrics["class_1_f1"]:.4f}')

    # Store feature importances
    fold_importance = pd.DataFrame({
        'feature':    feature_names,
        'importance': model.feature_importances_,
        'fold':       fold,
    })
    feature_importance_list.append(fold_importance)

    # Persist model artefacts
    joblib.dump(model, os.path.join(OUTPUT_DIR, f'rf_model_fold_{fold}.joblib'))
    with open(os.path.join(OUTPUT_DIR, f'rf_features_fold_{fold}.json'), 'w') as f:
        json.dump(feature_names, f)

    fold_result = {
        'fold':               fold,
        'n_features':         X_train.shape[1],
        'n_train_original':   len(y_train),
        'n_train_balanced':   len(y_train_bal),
        'n_val':              len(y_val),
        'n_test':             len(y_test),
        'n_val_eval':         len(y_val_bal),
        'n_test_eval':        len(y_test_bal),
        'eval_pos_ratio':     EVAL_CONFIG['eval_pos_ratio'],
        'imbalance_strategy': IMBALANCE_STRATEGY,
        'oob_score':          model.oob_score_ if hasattr(model, 'oob_score_') else None,
        'val':                val_metrics,
        'test':               test_metrics,
    }
    all_fold_results.append(fold_result)


# ===========================================================================
# SAVE AGGREGATED RESULTS
# ===========================================================================

importance_df   = pd.concat(feature_importance_list)
mean_importance = (
    importance_df
    .groupby('feature')['importance']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
top_n          = 50
top_importance = mean_importance.head(top_n)

summary_rows = []
for r in all_fold_results:
    row = {
        'fold':               r['fold'],
        'imbalance_strategy': r['imbalance_strategy'],
        'n_train_original':   r['n_train_original'],
        'n_train_balanced':   r['n_train_balanced'],
        'n_val_eval':         r['n_val_eval'],
        'n_test_eval':        r['n_test_eval'],
        'eval_pos_ratio':     r['eval_pos_ratio'],
        'oob_score':          r.get('oob_score'),
        # Validation metrics
        'val_macro_auc':      r['val']['macro_auc'],
        'val_macro_f1':       r['val']['macro_f1'],
        'val_weighted_f1':    r['val']['weighted_f1'],
        'val_log_loss':       r['val']['log_loss'],
        'val_balanced_acc':   r['val']['balanced_accuracy'],
        'val_kappa':          r['val']['cohen_kappa'],
        'val_mcc':            r['val']['mcc'],
        # Test metrics
        'test_macro_auc':     r['test']['macro_auc'],
        'test_macro_f1':      r['test']['macro_f1'],
        'test_weighted_f1':   r['test']['weighted_f1'],
        'test_log_loss':      r['test']['log_loss'],
        'test_balanced_acc':  r['test']['balanced_accuracy'],
        'test_kappa':         r['test']['cohen_kappa'],
        'test_mcc':           r['test']['mcc'],
    }
    for cls in range(2):
        for split in ('val', 'test'):
            row[f'{split}_class{cls}_auc']       = r[split].get(f'class_{cls}_auc',       np.nan)
            row[f'{split}_class{cls}_f1']        = r[split].get(f'class_{cls}_f1',        np.nan)
            row[f'{split}_class{cls}_recall']    = r[split].get(f'class_{cls}_recall',    np.nan)
            row[f'{split}_class{cls}_precision'] = r[split].get(f'class_{cls}_precision', np.nan)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

summary_path    = os.path.join(OUTPUT_DIR, 'household_rf_summary.csv')
importance_path = os.path.join(OUTPUT_DIR, f'household_rf_top{top_n}_features.csv')
full_json_path  = os.path.join(OUTPUT_DIR, 'household_rf_full_results.json')

summary_df.to_csv(summary_path, index=False)
top_importance.to_csv(importance_path, index=False)

with open(full_json_path, 'w') as f:
    json.dump({
        'imbalance_strategy': IMBALANCE_STRATEGY,
        'eval_config':        EVAL_CONFIG,
        'all_folds':          all_fold_results,
        'top_features':       top_importance.to_dict(orient='records'),
    }, f, indent=2)


# ===========================================================================
# PRINT CROSS-VALIDATION SUMMARY
# ===========================================================================

print('\n' + '=' * 80)
print(f'Random Forest — 5-Fold CV Summary  '
      f'(strategy: {IMBALANCE_STRATEGY.upper()})')
print(f'Evaluation: balanced {EVAL_CONFIG["eval_pos_ratio"]:.0%} positive, '
      f'n={EVAL_CONFIG["eval_size"]}')
print('=' * 80)

for split_name, prefix in [('VALIDATION', 'val'), ('TEST', 'test')]:
    print(f"\n{'='*60}")
    print(f'{split_name} — averaged across 5 folds')
    print(f"{'='*60}")
    for metric in ['macro_auc', 'macro_f1', 'weighted_f1', 'log_loss',
                   'balanced_acc', 'kappa', 'mcc']:
        col  = f'{prefix}_{metric}'
        mean = summary_df[col].mean()
        std  = summary_df[col].std()
        print(f'  {metric:<20}: {mean:.4f} ± {std:.4f}')
    for cls in range(2):
        print(f'\n  Class {cls} metrics:')
        for m in ('auc', 'f1', 'recall', 'precision'):
            col  = f'{prefix}_class{cls}_{m}'
            mean = summary_df[col].mean()
            std  = summary_df[col].std()
            print(f'    {m:<12}: {mean:.4f} ± {std:.4f}')

if 'oob_score' in summary_df and not summary_df['oob_score'].isna().all():
    print(f'\n  OOB Score: {summary_df["oob_score"].mean():.4f} '
          f'± {summary_df["oob_score"].std():.4f}')

print(f'\n{"="*80}')
print(f'Results saved to {OUTPUT_DIR}:')
print(f'  Summary CSV:      {summary_path}')
print(f'  Top {top_n} features: {importance_path}')
print(f'  Full JSON:        {full_json_path}')
print('=' * 80)
