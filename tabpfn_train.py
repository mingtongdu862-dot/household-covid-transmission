"""
tabpfn_train.py
===============
TabPFN ensemble training and evaluation script with 5-fold cross-validation.

This script is the direct counterpart to ``baseline_logistic_regression.py``,
``baseline_random_forest.py``, and ``baseline_xgboost.py``.  It trains the
TabPFN bagging ensemble described in the paper and evaluates it under the same
protocol used for all baselines: a balanced held-out subset drawn from the test
fold (n = 4,000, 50 % positive, without replacement).

Key design decisions
--------------------
- **Training context oversampling:** each of the *K* = 8 bags is sampled with a
  60 % positive ratio (24,000 positive + 16,000 negative), pushing the model's
  in-context prior toward the minority class and achieving Recall₊ = 0.905.
- **Shared evaluation set:** a single balanced eval sample is drawn once per
  fold and reused across all bags, making per-bag and ensemble metrics directly
  comparable.
- **Soft-voting:** the ensemble probability is the arithmetic mean of all bag
  predicted probabilities.
- **No data leakage:** the validation fold is kept strictly separate from the
  training pool; only ``train_fold_N`` is used for context.

Configuration is loaded from ``config.py``.

Outputs written to ``<OUTPUT_DIR>/improved_bagging_cv/``
--------------------------------------------------------
- ``fold_{k}_test_ensemble_metrics.json``   per-fold ensemble metric dict
- ``fold_{k}_full_results.json``            per-fold complete result dump
- ``fold_{k}_test_predictions.npy``         shape (n_bags, n_eval) probability array
- ``tabpfn_cv_summary.csv``                 mean ± std summary across 5 folds
- ``tabpfn_cv_full_results.json``           complete multi-fold result dump

Usage
-----
    python tabpfn_train.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix,
                             log_loss, balanced_accuracy_score,
                             cohen_kappa_score, matthews_corrcoef,
                             precision_recall_curve, auc)
import os
import json
import time
import torch
import warnings
from typing import Dict, Tuple, List
from tabpfn import TabPFNClassifier
warnings.filterwarnings('ignore')

# Import config
from config import *


# ===========================================================================
# TRAINING CONFIGURATION
# ===========================================================================
CV_CONFIG = {
    # Cross-validation
    'folds_to_run': [1, 2, 3, 4, 5],      # All 5 folds

    # Bagging configuration
    'n_bags': 8,                            # Number of independent bags
    'bag_train_size': 40000,                # Training samples per bag
    'bag_eval_size': 4000,                  # Val/Test samples per bag (each)

    # Class ratio configuration
    'train_pos_ratio': 0.6,                 # 60% positive in training bags
    'eval_pos_ratio':  0.5,                 # 50% positive in val/test bags

    # Ensemble
    'ensemble_method': 'soft_voting',

    # Random seed
    'random_state': 42,
}

TABPFN_MODEL_PARAMS = {
    'device': 'cuda',
    'n_estimators': 8,
    'ignore_pretraining_limits': False,
    'model_path': './tabpfn_weights/tabpfn-v2.5-classifier-v2.5_default.ckpt',
}


print("\n" + "="*80)
print("TabPFN IMPROVED BAGGING — 5-FOLD CROSS-VALIDATION")
print("="*80)
print(f"  Folds:            {CV_CONFIG['folds_to_run']}")
print(f"  Bags/fold:        {CV_CONFIG['n_bags']}")
print(f"  Train size/bag:   {CV_CONFIG['bag_train_size']:,}  ({CV_CONFIG['train_pos_ratio']*100:.0f}% positive)")
print(f"  Eval size/bag:    {CV_CONFIG['bag_eval_size']:,}  (50% positive)")
print(f"  Ensemble method:  {CV_CONFIG['ensemble_method']}")
print(f"  Device:           {TABPFN_MODEL_PARAMS['device']}")
print("="*80 + "\n")


# ===========================================================================
# DATA LOADING  (train / val / test kept separate — no leakage)
# ===========================================================================
def load_fold_data(fold: int):
    """
    Load one fold keeping train / val / test separate.
    Training pool = train_fold_N only (val is held out for evaluation).
    """
    base_path = FOLDS_PATH
    drop_cols  = DROP_COLS_BASE + DELETED_COLS
    label_col  = 'label'

    train_df = pd.read_csv(f'{base_path}/train_fold_{fold}.csv', encoding='latin1')
    val_df   = pd.read_csv(f'{base_path}/val_fold_{fold}.csv',   encoding='latin1')
    test_df  = pd.read_csv(f'{base_path}/test_fold_{fold}.csv',  encoding='latin1')

    for df in [train_df, val_df, test_df]:
        df.drop(columns=drop_cols, errors='ignore', inplace=True)
        df['label'] = (df['secondary_cases_count'] > 0).astype(int)
        df.drop('secondary_cases_count', axis=1, inplace=True)

    feature_names = [c for c in train_df.columns if c != label_col]

    for df in [train_df, val_df, test_df]:
        df.reset_index(drop=True, inplace=True)

    def _stats(df, name):
        n_pos = (df['label'] == 1).sum()
        return (f"{name}: {len(df):,} samples  "
                f"(pos {n_pos:,} / {n_pos/len(df):.1%},"
                f" neg {(df['label']==0).sum():,} / {(df['label']==0).mean():.1%})")

    print(f"\n{'='*80}")
    print(f"Fold {fold} — Data Loaded")
    print(f"{'='*80}")
    print(f"  {_stats(train_df, 'Train')}")
    print(f"  {_stats(val_df,   'Val  ')}")
    print(f"  {_stats(test_df,  'Test ')}")
    print(f"  Features: {len(feature_names):,}")

    return train_df, val_df, test_df, feature_names


# ===========================================================================
# SAMPLING
# ===========================================================================
def sample_with_ratio(df: pd.DataFrame, target_size: int, pos_ratio: float,
                      random_state: int = 42, tag: str = ''):
    """Sample data with a specific positive class ratio (without replacement)."""
    np.random.seed(random_state)

    n_pos_target = int(target_size * pos_ratio)
    n_neg_target = target_size - n_pos_target

    pos_samples = df[df['label'] == 1]
    neg_samples = df[df['label'] == 0]

    n_pos_avail = len(pos_samples)
    n_neg_avail = len(neg_samples)

    if n_pos_avail < n_pos_target:
        print(f"    WARNING [{tag}]: pos requested {n_pos_target:,} > available {n_pos_avail:,}")
        n_pos_target = n_pos_avail
        n_neg_target = target_size - n_pos_target

    if n_neg_avail < n_neg_target:
        print(f"    WARNING [{tag}]: neg requested {n_neg_target:,} > available {n_neg_avail:,}")
        n_neg_target = n_neg_avail
        n_pos_target = target_size - n_neg_target

    sampled = pd.concat([
        pos_samples.sample(n=n_pos_target, replace=False, random_state=random_state),
        neg_samples.sample(n=n_neg_target, replace=False, random_state=random_state),
    ])
    sampled = sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    actual_pos = sampled['label'].mean()
    print(f"    [{tag}] {len(sampled):,} samples  pos={actual_pos:.1%}")
    return sampled


# ===========================================================================
# METRICS  (matches XGBoost's compute_detailed_metrics output keys)
# ===========================================================================
def compute_metrics(y_true, y_pred, y_prob):
    """Return metric dict with the same key names as the XGBoost script."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)

    y_prob_2d = np.column_stack([1 - y_prob, y_prob])

    macro_auc  = roc_auc_score(y_true, y_prob)
    pr_auc     = average_precision_score(y_true, y_prob)

    result = {
        # ── Overall ──────────────────────────────────────────────────────
        'accuracy':          float((y_true == y_pred).mean()),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'cohen_kappa':       float(cohen_kappa_score(y_true, y_pred)),
        'mcc':               float(matthews_corrcoef(y_true, y_pred)),
        'log_loss':          float(log_loss(y_true, y_prob_2d)),

        # ── Macro ─────────────────────────────────────────────────────────
        'macro_auc':       float(macro_auc),
        'macro_pr_auc':    float(pr_auc),
        'macro_f1':        float(report['macro avg']['f1-score']),
        'macro_precision': float(report['macro avg']['precision']),
        'macro_recall':    float(report['macro avg']['recall']),
        'weighted_f1':     float(report['weighted avg']['f1-score']),

        # ── Confusion matrix ──────────────────────────────────────────────
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_tn': int(cm[0, 0]),
        'confusion_matrix_fp': int(cm[0, 1]),
        'confusion_matrix_fn': int(cm[1, 0]),
        'confusion_matrix_tp': int(cm[1, 1]),
    }

    # Per-class metrics (class 0 & 1)
    for cls in range(2):
        cls_key = str(cls)
        result[f'class_{cls}_precision'] = float(report.get(cls_key, {}).get('precision', np.nan))
        result[f'class_{cls}_recall']    = float(report.get(cls_key, {}).get('recall',    np.nan))
        result[f'class_{cls}_f1']        = float(report.get(cls_key, {}).get('f1-score',  np.nan))
        result[f'class_{cls}_support']   = int  (report.get(cls_key, {}).get('support',   0))

        try:
            if cls == 1:
                result[f'class_{cls}_auc']    = float(roc_auc_score(y_true, y_prob))
                result[f'class_{cls}_pr_auc'] = float(average_precision_score(y_true, y_prob))
            else:
                result[f'class_{cls}_auc']    = float(roc_auc_score((y_true==0).astype(int), 1-y_prob))
                prec0, rec0, _ = precision_recall_curve((y_true==0).astype(int), 1-y_prob)
                result[f'class_{cls}_pr_auc'] = float(auc(rec0, prec0))
        except Exception:
            result[f'class_{cls}_auc']    = float('nan')
            result[f'class_{cls}_pr_auc'] = float('nan')

    return result


def print_metrics(metrics: dict, title: str):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Macro AUC:        {metrics['macro_auc']:.4f}")
    print(f"Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:      {metrics['weighted_f1']:.4f}")
    print(f"Log Loss:         {metrics['log_loss']:.4f}")
    print(f"Balanced Acc:     {metrics['balanced_accuracy']:.4f}")
    print(f"Cohen Kappa:      {metrics['cohen_kappa']:.4f}")
    print(f"MCC:              {metrics['mcc']:.4f}")
    print(f"\nClass 1 (Has secondary) Metrics:")
    print(f"  AUC:            {metrics['class_1_auc']:.4f}")
    print(f"  PR-AUC:         {metrics['class_1_pr_auc']:.4f}")
    print(f"  Precision:      {metrics['class_1_precision']:.4f}")
    print(f"  Recall:         {metrics['class_1_recall']:.4f}")
    print(f"  F1:             {metrics['class_1_f1']:.4f}")
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                    Pred Neg   Pred Pos")
    print(f"  Actual Negative   {cm[0][0]:8d}   {cm[0][1]:8d}")
    print(f"  Actual Positive   {cm[1][0]:8d}   {cm[1][1]:8d}")


# ===========================================================================
# BAGGING ENSEMBLE  (for one fold, on one eval split)
# ===========================================================================
def run_bagging_ensemble(
    train_df:      pd.DataFrame,
    eval_df:       pd.DataFrame,
    feature_names: list,
    config:        dict,
    eval_tag:      str = 'Val',
) -> Tuple[Dict, List, np.ndarray]:
    """
    Train bagging ensemble on train_df, evaluate on eval_df.

    Returns
    -------
    ensemble_metrics  : dict of final aggregated metrics
    bag_metrics       : list of per-bag metric dicts
    all_proba         : np.ndarray shape (n_bags, n_eval_samples)
    """
    print(f"\n{'='*80}")
    print(f"Bagging Ensemble — evaluating on {eval_tag}")
    print(f"{'='*80}")

    # ── Sample eval set (shared across all bags) ──────────────────────────
    print(f"\n  Sampling {eval_tag} set (shared across bags):")
    eval_sample = sample_with_ratio(
        eval_df,
        config['bag_eval_size'],
        config['eval_pos_ratio'],
        random_state=config['random_state'] + 9999,
        tag=eval_tag,
    )
    X_eval = eval_sample[feature_names].values
    y_eval = eval_sample['label'].values

    # ── Per-bag loop ───────────────────────────────────────────────────────
    all_proba   = []
    bag_metrics = []
    train_idx_used = set()
    total_fit_time = 0.0

    for bag_id in range(config['n_bags']):
        print(f"\n  ── Bag {bag_id+1}/{config['n_bags']} ──")
        bag_seed = config['random_state'] + bag_id * 1000

        print(f"  Sampling training data:")
        train_sample = sample_with_ratio(
            train_df,
            config['bag_train_size'],
            config['train_pos_ratio'],
            random_state=bag_seed,
            tag=f"Train-bag{bag_id+1}",
        )
        train_idx_used.update(train_sample.index.tolist())

        X_train = train_sample[feature_names].values
        y_train = train_sample['label'].values

        # Train
        t0 = time.time()
        model = TabPFNClassifier(**TABPFN_MODEL_PARAMS)
        model.fit(X_train, y_train)
        fit_time = time.time() - t0
        total_fit_time += fit_time
        print(f"  ✓ Fit in {fit_time:.1f}s")

        # Predict
        y_proba = model.predict_proba(X_eval)[:, 1]
        y_pred  = (y_proba > 0.5).astype(int)
        all_proba.append(y_proba)

        bm = compute_metrics(y_eval, y_pred, y_proba)
        bm.update({'bag_id': bag_id+1, 'fit_time': fit_time})
        bag_metrics.append(bm)
        print(f"  Bag {bag_id+1}  macro_auc={bm['macro_auc']:.4f}  macro_f1={bm['macro_f1']:.4f}")

    # ── Ensemble via soft voting ───────────────────────────────────────────
    all_proba_arr   = np.array(all_proba)                   # (n_bags, n_eval)
    y_prob_ensemble = all_proba_arr.mean(axis=0)
    y_pred_ensemble = (y_prob_ensemble > 0.5).astype(int)

    ensemble_metrics = compute_metrics(y_eval, y_pred_ensemble, y_prob_ensemble)
    ensemble_metrics.update({
        'n_bags':             config['n_bags'],
        'bag_train_size':     config['bag_train_size'],
        'bag_eval_size':      config['bag_eval_size'],
        'train_pos_ratio':    config['train_pos_ratio'],
        'eval_pos_ratio':     config['eval_pos_ratio'],
        'ensemble_method':    config['ensemble_method'],
        'total_fit_time':     total_fit_time,
        'avg_fit_time':       total_fit_time / config['n_bags'],
        'data_coverage_pct':  len(train_idx_used) / len(train_df) * 100,
    })

    # Improvement summary
    bag_aucs = [m['macro_auc'] for m in bag_metrics]
    bag_f1s  = [m['macro_f1']  for m in bag_metrics]
    print(f"\n  Ensemble vs Bags ({eval_tag}):")
    print(f"    macro_auc  ensemble={ensemble_metrics['macro_auc']:.4f}  "
          f"best_bag={max(bag_aucs):.4f}  avg={np.mean(bag_aucs):.4f}±{np.std(bag_aucs):.4f}")
    print(f"    macro_f1   ensemble={ensemble_metrics['macro_f1']:.4f}  "
          f"best_bag={max(bag_f1s):.4f}  avg={np.mean(bag_f1s):.4f}±{np.std(bag_f1s):.4f}")

    return ensemble_metrics, bag_metrics, all_proba_arr


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("\n" + "="*80)
    print("TabPFN IMPROVED BAGGING — 5-FOLD CROSS-VALIDATION")
    print("="*80)

    # Device check
    if TABPFN_MODEL_PARAMS['device'] == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        TABPFN_MODEL_PARAMS['device'] = 'cpu'
    if TABPFN_MODEL_PARAMS['device'] == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB\n")

    output_dir = os.path.join(OUTPUT_DIR, 'improved_bagging_cv')
    os.makedirs(output_dir, exist_ok=True)

    all_fold_results = []     # list of dicts, one per fold

    # ─────────────────────────────────────────────────────────────────────
    for fold in CV_CONFIG['folds_to_run']:
        print(f"\n{'#'*80}")
        print(f"# FOLD {fold}")
        print(f"{'#'*80}")

        try:
            train_df, val_df, test_df, feature_names = load_fold_data(fold)

            # ── Evaluate on VAL set (commented out to save compute) ──────
            # val_ens, val_bags, val_proba = run_bagging_ensemble(
            #     train_df, val_df, feature_names, CV_CONFIG, eval_tag='Val'
            # )
            # print_metrics(val_ens, title=f"FOLD {fold} — VAL SET RESULTS")

            # ── Evaluate on TEST set ──────────────────────────────────────
            test_ens, test_bags, test_proba = run_bagging_ensemble(
                train_df, test_df, feature_names, CV_CONFIG, eval_tag='Test'
            )
            print_metrics(test_ens, title=f"FOLD {fold} — TEST SET RESULTS")

            # ── Save per-fold artefacts ───────────────────────────────────
            fold_result = {
                'fold': fold,
                'n_features':  len(feature_names),
                'n_train':     len(train_df),
                'n_val':       len(val_df),
                'n_test':      len(test_df),
                'test': test_ens,
                'test_bag_metrics': test_bags,
            }

            # val artefacts skipped (val evaluation disabled)
            with open(os.path.join(output_dir, f'fold_{fold}_test_ensemble_metrics.json'), 'w') as f:
                json.dump(test_ens, f, indent=2)
            with open(os.path.join(output_dir, f'fold_{fold}_full_results.json'), 'w') as f:
                json.dump(fold_result, f, indent=2)
            np.save(os.path.join(output_dir, f'fold_{fold}_test_predictions.npy'), test_proba)

            all_fold_results.append(fold_result)
            print(f"\n✓ Fold {fold} complete — results saved to {output_dir}")

        except Exception as e:
            print(f"\n✗ ERROR fold {fold}: {e}")
            import traceback; traceback.print_exc()
            continue

    # ─────────────────────────────────────────────────────────────────────
    # Build summary CSV (same structure as XGBoost summary)
    # ─────────────────────────────────────────────────────────────────────
    if not all_fold_results:
        print("No results to summarise.")
        return []

    summary_rows = []
    for r in all_fold_results:
        row = {
            'fold':       r['fold'],
            'n_train':    r['n_train'],
            'n_test':     r['n_test'],
            'n_features': r['n_features'],
        }
        m = r['test']
        row.update({
            'test_macro_auc':       m['macro_auc'],
            'test_macro_f1':        m['macro_f1'],
            'test_weighted_f1':     m['weighted_f1'],
            'test_log_loss':        m['log_loss'],
            'test_balanced_acc':    m['balanced_accuracy'],
            'test_cohen_kappa':     m['cohen_kappa'],
            'test_mcc':             m['mcc'],
        })
        for cls in range(2):
            row[f'test_class{cls}_auc']       = m.get(f'class_{cls}_auc',       float('nan'))
            row[f'test_class{cls}_pr_auc']    = m.get(f'class_{cls}_pr_auc',    float('nan'))
            row[f'test_class{cls}_f1']        = m.get(f'class_{cls}_f1',        float('nan'))
            row[f'test_class{cls}_recall']    = m.get(f'class_{cls}_recall',    float('nan'))
            row[f'test_class{cls}_precision'] = m.get(f'class_{cls}_precision', float('nan'))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, 'tabpfn_cv_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    # Full JSON dump
    full_json_path = os.path.join(output_dir, 'tabpfn_cv_full_results.json')
    with open(full_json_path, 'w') as f:
        json.dump(all_fold_results, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────
    # Console summary — mirrors XGBoost print format exactly
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("TabPFN IMPROVED BAGGING — 5-FOLD CROSS-VALIDATION RESULTS")
    print("="*80)

    print(f"\n{'='*60}")
    print(f"TEST SET — AVERAGED ACROSS {len(summary_df)} FOLDS:")
    print(f"{'='*60}")
    print(f"Macro AUC:           {summary_df['test_macro_auc'].mean():.4f} ± {summary_df['test_macro_auc'].std():.4f}")
    print(f"Macro F1:            {summary_df['test_macro_f1'].mean():.4f} ± {summary_df['test_macro_f1'].std():.4f}")
    print(f"Weighted F1:         {summary_df['test_weighted_f1'].mean():.4f} ± {summary_df['test_weighted_f1'].std():.4f}")
    print(f"Log Loss:            {summary_df['test_log_loss'].mean():.4f} ± {summary_df['test_log_loss'].std():.4f}")
    print(f"Balanced Accuracy:   {summary_df['test_balanced_acc'].mean():.4f} ± {summary_df['test_balanced_acc'].std():.4f}")
    print(f"Cohen Kappa:         {summary_df['test_cohen_kappa'].mean():.4f} ± {summary_df['test_cohen_kappa'].std():.4f}")
    print(f"MCC:                 {summary_df['test_mcc'].mean():.4f} ± {summary_df['test_mcc'].std():.4f}")
    print(f"\nClass 1 (Has secondary) Metrics:")
    print(f"  AUC:               {summary_df['test_class1_auc'].mean():.4f} ± {summary_df['test_class1_auc'].std():.4f}")
    print(f"  PR-AUC:            {summary_df['test_class1_pr_auc'].mean():.4f} ± {summary_df['test_class1_pr_auc'].std():.4f}")
    print(f"  F1:                {summary_df['test_class1_f1'].mean():.4f} ± {summary_df['test_class1_f1'].std():.4f}")
    print(f"  Recall:            {summary_df['test_class1_recall'].mean():.4f} ± {summary_df['test_class1_recall'].std():.4f}")
    print(f"  Precision:         {summary_df['test_class1_precision'].mean():.4f} ± {summary_df['test_class1_precision'].std():.4f}")

    print(f"\n{'='*80}")
    print(f"Results saved to:  {output_dir}")
    print(f"  • Summary CSV    : {summary_path}")
    print(f"  • Full JSON      : {full_json_path}")
    print("="*80 + "\n")

    return all_fold_results


if __name__ == "__main__":
    results = main()