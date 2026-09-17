"""
TabPFN Ensemble - Training + Inference + Explainability Analysis (V4 + Checkpoint Resume)
======================================================================

Changelog (v4-resume → v4-resume-top50):
  - Added pi_top_n_features config key to SUBGROUP_CONFIG
  - Subgroup PI now only evaluates the top-N globally ranked features (default: 50)
  - Time estimate: 295 → 50 features reduces subgroup PI cost by ~83%,
    total runtime drops from ~82h to ~16h
  - All other logic (Beeswarm, Local SHAP, checkpointing) is unchanged
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix,
                             log_loss, balanced_accuracy_score,
                             cohen_kappa_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split
import shap
import os
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import time
import torch
import warnings
warnings.filterwarnings('ignore')

from config import *
from tabpfn_ensemble import TabPFNEnsemble

# ===========================================================================
# ANALYSIS CONFIGURATION
# ===========================================================================

GLOBAL_PI_CONFIG = {
    'subset_size':   3000,
    'n_repeats':     5,
    'top_n_display': 30,
}

GLOBAL_BEESWARM_CONFIG = {
    'top_k_features': 10,
    'n_quartiles':     4,
    'n_per_cell':      5,
    'n_background':   30,
    'max_evals':     100,
    'batch_size':     50,
}

LOCAL_SHAP_CONFIG = {
    'n_per_outcome':     4,
    'n_per_feat_value':  2,
    'n_top_features':    5,
    'max_evals':         100,
    'n_background':      30,
    'batch_size':        50,
    'max_waterfall':     15,
}

SUBGROUP_CONFIG = {
    'pi_subset_size':       1000,
    'pi_n_repeats':         3,
    'top_n_display':        20,
    'pi_top_n_features':    50,    # NEW: subgroup PI only evaluates the global top-N features
    'beeswarm_top_k':         5,
    'beeswarm_n_quartiles':   4,
    'beeswarm_n_per_cell':    4,
    'beeswarm_n':           100,
    'beeswarm_max_evals':   100,
    'beeswarm_n_bg':         30,
    'local_n_per_label':      3,
    'local_n_per_feat':       2,
    'local_n_top_feats':      3,
    'local_max_evals':       100,
    'local_n_background':    30,
    'min_size':              30,
}


# ===========================================================================
# REMAINING TIME ESTIMATION
# ===========================================================================

def estimate_remaining_time(feature_names, n_subgroups=11,
                            global_pi_done=False, global_bee_batches_done=0,
                            global_bee_total_batches=0):
    """
    Estimate remaining wall-clock time per stage based on empirical per-call timings.
    Subgroup PI estimate accounts for the top-N feature filter in SUBGROUP_CONFIG.
    """
    n_feat = len(feature_names)
    min_per_call_3k = 2129.0 / (n_feat * GLOBAL_PI_CONFIG['n_repeats'])
    scale_1k = 1000.0 / 3000.0

    bee_samples = (GLOBAL_BEESWARM_CONFIG['top_k_features'] *
                   GLOBAL_BEESWARM_CONFIG['n_quartiles'] * 2 *
                   GLOBAL_BEESWARM_CONFIG['n_per_cell'])
    bee_total_batches = (bee_samples + GLOBAL_BEESWARM_CONFIG['batch_size'] - 1) \
                        // GLOBAL_BEESWARM_CONFIG['batch_size']
    min_per_bee_batch = 5.0

    print(f"\n{'='*80}")
    print("⏱  Remaining Time Estimation")
    print(f"{'='*80}")
    print(f"  Baseline: per model call ({3000} samples) ≈ {min_per_call_3k:.2f} min")
    print(f"            scaled to 1000 samples ≈ {min_per_call_3k*scale_1k:.2f} min/call\n")

    total_min = 0.0

    # Global PI
    if not global_pi_done:
        pi_min = n_feat * GLOBAL_PI_CONFIG['n_repeats'] * min_per_call_3k
        print(f"  Global PI           : {pi_min:.0f} min  ({pi_min/60:.1f} h)")
        total_min += pi_min
    else:
        print(f"  Global PI           : ✅ Already done (skipped)")

    # Global Beeswarm
    bee_remaining = max(0, bee_total_batches - global_bee_batches_done)
    bee_min = bee_remaining * min_per_bee_batch
    status = f"({global_bee_batches_done}/{bee_total_batches} batches done, {bee_remaining} remaining)"
    print(f"  Global Beeswarm     : {bee_min:.0f} min  {status}")
    total_min += bee_min

    # Global local SHAP
    local_samples = (LOCAL_SHAP_CONFIG['n_per_outcome'] * 4 +
                     LOCAL_SHAP_CONFIG['n_per_feat_value'] * 2 *
                     LOCAL_SHAP_CONFIG['n_top_features'] * 2)
    local_batches = (local_samples + LOCAL_SHAP_CONFIG['batch_size'] - 1) \
                    // LOCAL_SHAP_CONFIG['batch_size']
    local_min = local_batches * min_per_bee_batch
    print(f"  Global Local SHAP   : {local_min:.0f} min  (~{local_samples} samples)")
    total_min += local_min

    # Subgroup analysis — use filtered feature count for PI estimate
    n_sg_feat = SUBGROUP_CONFIG.get('pi_top_n_features', n_feat)
    sg_pi_calls_per_sg = n_sg_feat * SUBGROUP_CONFIG['pi_n_repeats']
    sg_pi_min_per_sg = sg_pi_calls_per_sg * min_per_call_3k * scale_1k
    sg_bee_batches = (SUBGROUP_CONFIG['beeswarm_n'] + 49) // 50
    sg_bee_min_per_sg = sg_bee_batches * min_per_bee_batch
    sg_local_min_per_sg = 2 * min_per_bee_batch
    sg_total_per_sg = sg_pi_min_per_sg + sg_bee_min_per_sg + sg_local_min_per_sg
    sg_total = sg_total_per_sg * n_subgroups

    print(f"  Subgroup PI features: {n_sg_feat} (top-N filtered from {n_feat} total)")
    print(f"  Subgroup PI (each)  : {sg_pi_min_per_sg:.0f} min  ({sg_pi_min_per_sg/60:.1f} h)")
    print(f"  Subgroup Beeswarm   : {sg_bee_min_per_sg:.0f} min/group")
    print(f"  Subgroup total ({n_subgroups}): {sg_total:.0f} min  ({sg_total/60:.1f} h)")
    total_min += sg_total

    print(f"\n  {'─'*50}")
    print(f"  📌 Total remaining  : {total_min:.0f} min  ({total_min/60:.1f} h)")
    print(f"  📌 Recommended node : {total_min/60*1.2:.0f} h  (+20% buffer)")
    print(f"{'='*80}\n")
    return total_min


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def load_and_preprocess(fold: int):
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

    pool_df = pd.concat([train_df, val_df], ignore_index=True)
    X_train = pool_df[feature_names]
    y_train = pool_df[label_col].values

    # Use the full held-out test fold as-is (same 25,248-household test set
    # evaluated in tabpfn_train.py's Table 2). Earlier versions drew a further
    # TEST_SAMPLE_RATIO-sized stratified subsample here, which silently
    # evaluated XAI on a different, smaller test set than the one reported
    # for predictive performance.
    X_test = test_df[feature_names]
    y_test = test_df[label_col].values

    print(f"  Train : {len(X_train):,} | {len(feature_names):,} features")
    print(f"  Test  : {len(X_test):,}")
    print(f"  Pos rate (train): {y_train.mean()*100:.1f}%")
    return X_train, y_train, X_test, y_test, feature_names


def compute_detailed_metrics(y_true, y_pred, y_prob):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        'accuracy':           float((y_true == y_pred).mean()),
        'roc_auc':            float(roc_auc_score(y_true, y_prob)),
        'macro_f1':           float(report.get('macro avg', {}).get('f1-score', np.nan)),
        'weighted_f1':        float(report.get('weighted avg', {}).get('f1-score', np.nan)),
        'log_loss':           float(log_loss(y_true, np.column_stack([1-y_prob, y_prob]))),
        'balanced_accuracy':  float(balanced_accuracy_score(y_true, y_pred)),
        'cohen_kappa':        float(cohen_kappa_score(y_true, y_pred)),
        'mcc':                float(matthews_corrcoef(y_true, y_pred)),
        'class_1_auc':        roc_auc_score(y_true, y_prob),
        'class_1_pr_auc':     average_precision_score(y_true, y_prob),
        'class_1_f1':         float(report.get('1', {}).get('f1-score', np.nan)),
        'class_1_recall':     float(report.get('1', {}).get('recall', np.nan)),
        'class_1_precision':  float(report.get('1', {}).get('precision', np.nan)),
        'confusion_matrix':   confusion_matrix(y_true, y_pred).tolist(),
    }


# ===========================================================================
# PERMUTATION IMPORTANCE
# ===========================================================================

def compute_permutation_importance(model, X, y, feature_names,
                                   n_repeats=5, subset_size=None,
                                   random_state=42):
    rng = np.random.default_rng(random_state)

    if subset_size is not None and subset_size < len(X):
        idx = rng.choice(len(X), size=subset_size, replace=False)
        X_use = X.iloc[idx].reset_index(drop=True)
        y_use = y[idx]
        print(f"    PI subset: {len(X_use):,} / {len(X):,}")
    else:
        X_use = X.reset_index(drop=True)
        y_use = y

    baseline_prob = model.predict_proba(X_use)[:, 1]
    baseline_auc  = roc_auc_score(y_use, baseline_prob)
    print(f"    Baseline AUC: {baseline_auc:.4f}")

    n_features = len(feature_names)
    all_drops  = np.zeros((n_features, n_repeats))

    t0 = time.time()
    for fi, feat in enumerate(feature_names):
        for r in range(n_repeats):
            X_perm = X_use.copy()
            X_perm[feat] = rng.permutation(X_perm[feat].values)
            perm_prob = model.predict_proba(X_perm)[:, 1]
            perm_auc  = roc_auc_score(y_use, perm_prob)
            all_drops[fi, r] = baseline_auc - perm_auc

        elapsed = time.time() - t0
        eta = elapsed / (fi + 1) * (n_features - fi - 1)
        print(f"\r    [{fi+1:3d}/{n_features}] {feat[:35]:35s} "
              f"mean_drop={all_drops[fi].mean():.4f}  "
              f"ETA {eta/60:.1f} min", end='', flush=True)
    print()

    importance_df = pd.DataFrame({
        'feature':          feature_names,
        'mean_importance':  all_drops.mean(axis=1),
        'std_importance':   all_drops.std(axis=1),
    }).sort_values('mean_importance', ascending=False).reset_index(drop=True)
    importance_df['rank'] = np.arange(1, len(importance_df) + 1)
    return importance_df


# ===========================================================================
# SHAP WITH BATCH-LEVEL CHECKPOINTS
# ===========================================================================

def compute_shap_small_with_checkpoint(model, X_explain, X_background,
                                       feature_names, checkpoint_dir,
                                       prefix='shap',
                                       max_evals=100, batch_size=50):
    """
    KernelExplainer SHAP with per-batch checkpointing.
    Each batch is saved as {checkpoint_dir}/{prefix}_batch_{i}.npy.
    Completed batches are detected at startup and skipped automatically.

    Returns shap_matrix: (len(X_explain), len(feature_names))
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    n           = len(X_explain)
    n_batch     = (n + batch_size - 1) // batch_size
    batch_files = [os.path.join(checkpoint_dir, f'{prefix}_batch_{i}.npy')
                   for i in range(n_batch)]

    # Detect how many batches are already complete
    first_pending = 0
    for i, fp in enumerate(batch_files):
        if os.path.exists(fp):
            first_pending = i + 1
        else:
            break

    if first_pending == n_batch:
        print(f"    ✅ SHAP checkpoints complete ({n_batch} batches), loading directly")
        return np.vstack([np.load(fp) for fp in batch_files])

    if first_pending > 0:
        print(f"    ♻️  Resuming from batch {first_pending+1}/{n_batch} "
              f"({first_pending} batches already done)")

    print(f"    SHAP: {n} samples × {max_evals} evals × {len(X_background)} background")

    predict_fn = lambda x: model.predict_proba(x)[:, 1]
    explainer  = shap.KernelExplainer(predict_fn, X_background, link='identity')

    t0 = time.time()
    for i in range(first_pending, n_batch):
        start   = i * batch_size
        end     = min(start + batch_size, n)
        X_batch = X_explain.iloc[start:end]
        pct     = (i + 1) / n_batch * 100
        print(f"\r      batch [{i+1}/{n_batch}] {pct:.0f}%", end='', flush=True)

        sv = explainer.shap_values(X_batch, nsamples=max_evals, silent=True)
        np.save(batch_files[i], sv)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n    → New batches complete, elapsed {elapsed/60:.1f} min")

    return np.vstack([np.load(fp) for fp in batch_files])


def _quartile_label_sample(X, y, pi_feature_order, feature_names,
                           top_k=10, n_quartiles=4, n_per_cell=5,
                           random_state=42):
    """
    Build a stratified explanation pool by crossing feature quartile buckets
    with class labels. Returns a representative subset for beeswarm SHAP.
    """
    rng      = np.random.default_rng(random_state)
    X_reset  = X.reset_index(drop=True)
    chosen   = set()
    top_feats = [f for f in pi_feature_order if f in X_reset.columns][:top_k]

    for feat in top_feats:
        vals = X_reset[feat].values.astype(float)
        boundaries = np.unique(
            np.nanpercentile(vals, np.linspace(0, 100, n_quartiles + 1)))
        if len(boundaries) < 2:
            boundaries = np.array([vals.min()-1e-9, np.median(vals), vals.max()+1e-9])
        buckets = np.digitize(vals, boundaries[1:-1])
        for b in range(len(boundaries) - 1):
            b_mask = buckets == b
            for lab in np.unique(y):
                lab_mask = y == lab
                cell_idx = np.where(b_mask & lab_mask)[0]
                if len(cell_idx) == 0:
                    continue
                k = min(n_per_cell, len(cell_idx))
                chosen.update(rng.choice(cell_idx, k, replace=False).tolist())

    chosen = sorted(chosen)
    print(f"    Beeswarm stratified pool: {len(chosen)} samples "
          f"(top-{top_k} features × {n_quartiles} quartiles × 2 labels × {n_per_cell}/cell)")
    return X_reset.iloc[chosen].reset_index(drop=True), y[chosen]


# ===========================================================================
# GLOBAL ANALYSIS WITH CHECKPOINT RESUME
# ===========================================================================

def global_feature_analysis(model, X_test, y_test, feature_names,
                             pi_cfg=GLOBAL_PI_CONFIG,
                             beeswarm_cfg=GLOBAL_BEESWARM_CONFIG,
                             save_dir=None):
    print(f"\n{'='*80}")
    print("GLOBAL FEATURE ANALYSIS")
    print(f"{'='*80}")

    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, 'global_importance')
    os.makedirs(save_dir, exist_ok=True)

    top_n         = pi_cfg['top_n_display']
    pi_path       = os.path.join(save_dir, 'pi_feature_importance.csv')
    bee_path      = os.path.join(save_dir, 'beeswarm.png')
    bee_shap_path = os.path.join(save_dir, 'beeswarm_shap_values.csv')
    ckpt_dir      = os.path.join(save_dir, 'shap_checkpoints')

    # ------------------------------------------------------------------ #
    # 1. Permutation Importance  — skip if CSV checkpoint exists          #
    # ------------------------------------------------------------------ #
    if os.path.exists(pi_path):
        print(f"\n  [1/2] ✅ PI checkpoint found, loading: {pi_path}")
        pi_df = pd.read_csv(pi_path)
    else:
        print(f"\n  [1/2] Permutation Importance...")
        t0 = time.time()
        pi_df = compute_permutation_importance(
            model, X_test, y_test, feature_names,
            n_repeats=pi_cfg['n_repeats'],
            subset_size=pi_cfg['subset_size'])
        pi_df.to_csv(pi_path, index=False)
        print(f"  PI done in {(time.time()-t0)/60:.1f} min")

    # Redraw PI bar chart (idempotent)
    top_pi = pi_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    ax.barh(range(len(top_pi)), top_pi['mean_importance'],
            xerr=top_pi['std_importance'],
            color='steelblue', ecolor='gray', capsize=3, alpha=0.85)
    ax.set_yticks(range(len(top_pi)))
    ax.set_yticklabels(top_pi['feature'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean AUC drop (permutation)', fontsize=11)
    ax.set_title(f'Global Feature Importance — Top {top_n}\n'
                 f'(PI, n={pi_cfg["subset_size"]:,}, {pi_cfg["n_repeats"]} repeats)',
                 fontsize=12, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pi_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------ #
    # 2. Beeswarm SHAP  — resume from batch-level checkpoints             #
    # ------------------------------------------------------------------ #
    if os.path.exists(bee_path) and os.path.exists(bee_shap_path):
        print(f"\n  [2/2] ✅ Beeswarm already exists, skipping")
        shap_matrix = pd.read_csv(bee_shap_path).values
        pi_order    = pi_df['feature'].tolist()
        X_explain, y_explain = _quartile_label_sample(
            X_test, y_test, pi_feature_order=pi_order,
            feature_names=feature_names,
            top_k=beeswarm_cfg['top_k_features'],
            n_quartiles=beeswarm_cfg['n_quartiles'],
            n_per_cell=beeswarm_cfg['n_per_cell'], random_state=1)
    else:
        print(f"\n  [2/2] Quartile×label stratified SHAP → Beeswarm...")
        pi_order   = pi_df['feature'].tolist()
        feat_index = {f: i for i, f in enumerate(feature_names)}

        try:
            bg_idx, _, _, _ = train_test_split(
                np.arange(len(X_test)), y_test,
                train_size=beeswarm_cfg['n_background'],
                stratify=y_test, random_state=0)
        except Exception:
            bg_idx = np.random.default_rng(0).choice(
                len(X_test), beeswarm_cfg['n_background'], replace=False)
        X_bg = X_test.reset_index(drop=True).iloc[bg_idx].reset_index(drop=True)

        X_explain, y_explain = _quartile_label_sample(
            X_test, y_test, pi_feature_order=pi_order,
            feature_names=feature_names,
            top_k=beeswarm_cfg['top_k_features'],
            n_quartiles=beeswarm_cfg['n_quartiles'],
            n_per_cell=beeswarm_cfg['n_per_cell'], random_state=1)

        t0 = time.time()
        shap_matrix = compute_shap_small_with_checkpoint(
            model, X_explain, X_bg, feature_names,
            checkpoint_dir=ckpt_dir, prefix='global_bee',
            max_evals=beeswarm_cfg['max_evals'],
            batch_size=beeswarm_cfg['batch_size'])
        print(f"  Beeswarm SHAP done in {(time.time()-t0)/60:.1f} min")

        pd.DataFrame(shap_matrix, columns=feature_names).to_csv(bee_shap_path, index=False)

        ordered_idx   = [feat_index[f] for f in pi_order[:top_n] if f in feat_index]
        ordered_names = [feature_names[i] for i in ordered_idx]
        shap_reordered = shap_matrix[:, ordered_idx]
        X_reordered    = X_explain[ordered_names]

        fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.45)))
        plt.sca(ax)
        shap.summary_plot(
            shap_reordered, X_reordered,
            feature_names=ordered_names, max_display=top_n,
            plot_type='dot', show=False)
        ax = plt.gca()
        ax.set_title(
            f'Beeswarm — SHAP values  '
            f'(n={len(X_explain)}, quartile×label stratified)\n'
            f'Features ordered by Permutation Importance rank',
            fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(bee_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Beeswarm saved.")

    # Combined summary table
    shap_importance = pd.DataFrame({
        'feature':       feature_names,
        'mean_abs_shap': np.abs(shap_matrix).mean(axis=0),
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    summary = pi_df.merge(
        shap_importance.rename(columns={'mean_abs_shap': 'mean_abs_shap_stratified'}),
        on='feature', how='left')
    summary.to_csv(os.path.join(save_dir, 'global_importance_summary.csv'), index=False)

    print(f"\n  Top 10 features (PI rank):")
    for _, row in pi_df.head(10).iterrows():
        print(f"    {int(row['rank']):2d}. {row['feature']:50s} | {row['mean_importance']:.4f}")

    return pi_df, shap_matrix, X_explain


# ===========================================================================
# LOCAL SHAP
# ===========================================================================

def _collect_local_indices(y_test, y_pred, y_prob, feature_vals,
                           top_feature_names, cfg):
    """
    Select a representative set of sample indices for local SHAP waterfall plots.
    Covers TP/FP/TN/FN outcomes as well as high/low values of the top features.
    """
    rng    = np.random.default_rng(42)
    chosen = set()

    def pick(mask, n):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return []
        k = min(n, len(idx))
        return rng.choice(idx, k, replace=False).tolist()

    # One sample per confusion-matrix cell
    n  = cfg['n_per_outcome']
    tp = (y_pred == 1) & (y_test == 1)
    fp = (y_pred == 1) & (y_test == 0)
    tn = (y_pred == 0) & (y_test == 0)
    fn = (y_pred == 0) & (y_test == 1)
    for mask in [tp, fp, tn, fn]:
        chosen.update(pick(mask, n))

    # Samples split by above/below-median feature value × label
    n2      = cfg['n_per_feat_value']
    medians = feature_vals.median()
    for feat in top_feature_names[:cfg['n_top_features']]:
        if feat not in feature_vals.columns:
            continue
        high = feature_vals[feat] >= medians[feat]
        low  = ~high
        for val_mask in [high, low]:
            for lab in [0, 1]:
                lab_mask = y_test == lab
                combined = val_mask.values & lab_mask
                chosen.update(pick(combined, n2))

    return sorted(chosen)


def local_shap_analysis(model, X_test, y_test, y_pred, y_prob, feature_names,
                        top_feature_names, cfg=LOCAL_SHAP_CONFIG, save_dir=None):
    print(f"\n{'='*80}")
    print("LOCAL SHAP ANALYSIS")
    print(f"{'='*80}")

    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, 'local_shap')
    os.makedirs(save_dir, exist_ok=True)

    npy_path = os.path.join(save_dir, 'local_shap_values.npy')
    idx_path = os.path.join(save_dir, 'local_shap_indices.npy')
    ckpt_dir = os.path.join(save_dir, 'shap_checkpoints')

    indices = _collect_local_indices(
        y_test, y_pred, y_prob, X_test, top_feature_names, cfg)
    print(f"  Selected {len(indices)} samples for local SHAP")

    X_local = X_test.iloc[indices].reset_index(drop=True)
    y_local = y_test[indices]

    # Load from checkpoint if indices match
    if os.path.exists(npy_path) and os.path.exists(idx_path):
        saved_idx = np.load(idx_path).tolist()
        if saved_idx == indices:
            print(f"  ✅ Local SHAP checkpoint found, loading directly")
            shap_matrix = np.load(npy_path)
        else:
            print(f"  ⚠️  Index mismatch, recomputing")
            os.makedirs(ckpt_dir, exist_ok=True)
            bg_idx = np.random.default_rng(0).choice(
                len(X_test), cfg['n_background'], replace=False)
            X_bg = X_test.iloc[bg_idx].reset_index(drop=True)
            shap_matrix = compute_shap_small_with_checkpoint(
                model, X_local, X_bg, feature_names,
                checkpoint_dir=ckpt_dir, prefix='local',
                max_evals=cfg['max_evals'], batch_size=cfg['batch_size'])
            np.save(npy_path, shap_matrix)
            np.save(idx_path, np.array(indices))
    else:
        bg_idx = np.random.default_rng(0).choice(
            len(X_test), cfg['n_background'], replace=False)
        X_bg = X_test.iloc[bg_idx].reset_index(drop=True)
        t0 = time.time()
        shap_matrix = compute_shap_small_with_checkpoint(
            model, X_local, X_bg, feature_names,
            checkpoint_dir=ckpt_dir, prefix='local',
            max_evals=cfg['max_evals'], batch_size=cfg['batch_size'])
        np.save(npy_path, shap_matrix)
        np.save(idx_path, np.array(indices))
        print(f"  Local SHAP done in {(time.time()-t0)/60:.1f} min")

    # Build JSON explanations and waterfall plots
    explanations = []
    for i, orig_idx in enumerate(indices):
        shap_vals = shap_matrix[i]
        feat_shap = sorted(zip(feature_names, shap_vals),
                           key=lambda x: x[1], reverse=True)
        y_pred_i = int(y_pred[orig_idx])
        y_true_i = int(y_test[orig_idx])
        outcome  = {(1, 1): 'TP', (1, 0): 'FP',
                    (0, 0): 'TN', (0, 1): 'FN'}.get((y_pred_i, y_true_i), '?')
        explanations.append({
            'sample_index': int(orig_idx), 'outcome': outcome,
            'prediction':   y_pred_i, 'probability': float(y_prob[orig_idx]),
            'true_label':   y_true_i,
            'top_positive': {f: float(s) for f, s in feat_shap[:5]},
            'top_negative': {f: float(s) for f, s in feat_shap[-5:]},
        })

    with open(os.path.join(save_dir, 'local_explanations.json'), 'w') as fh:
        json.dump(explanations, fh, indent=2, ensure_ascii=False)

    plotted = {}
    for i, exp in enumerate(explanations):
        out   = exp['outcome']
        fname = f'waterfall_{out}_sample{exp["sample_index"]}.png'
        fpath = os.path.join(save_dir, fname)
        if os.path.exists(fpath):
            plotted[out] = plotted.get(out, 0) + 1
            continue
        fig, _ = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_matrix[i],
                base_values=float(y_prob.mean()),
                data=X_local.iloc[i].values,
                feature_names=feature_names),
            max_display=cfg['max_waterfall'], show=False)
        plt.title(f"{out} | prob={exp['probability']:.3f} | true={exp['true_label']}",
                  fontsize=11)
        plt.tight_layout()
        plt.savefig(fpath, dpi=300, bbox_inches='tight')
        plt.close()
        plotted[out] = plotted.get(out, 0) + 1

    print(f"  Waterfall: {dict(plotted)}")
    print(f"  ✓ Local SHAP complete")
    return explanations


# ===========================================================================
# SUBGROUP ANALYSIS WITH CHECKPOINT RESUME
# ===========================================================================

def _define_subgroups(y_test, y_pred, y_prob):
    return {
        'high_risk':      y_prob >= 0.7,
        'moderate_risk':  (y_prob >= 0.3) & (y_prob < 0.7),
        'low_risk':       y_prob < 0.3,
        'predicted_pos':  y_pred == 1,
        'predicted_neg':  y_pred == 0,
        'true_positive':  (y_pred == 1) & (y_test == 1),
        'true_negative':  (y_pred == 0) & (y_test == 0),
        'false_positive': (y_pred == 1) & (y_test == 0),
        'false_negative': (y_pred == 0) & (y_test == 1),
        'correct':        y_pred == y_test,
        'incorrect':      y_pred != y_test,
    }


def subgroup_analysis(model, X_test, y_test, y_pred, y_prob, feature_names,
                      cfg=SUBGROUP_CONFIG, save_dir=None,
                      global_pi_df=None):
    """
    Run per-subgroup PI, Beeswarm SHAP, and local SHAP with checkpoint resume.

    Parameters
    ----------
    global_pi_df : pd.DataFrame, optional
        Global permutation importance results with a 'feature' column sorted
        by descending importance. When provided, subgroup PI is restricted to
        the top cfg['pi_top_n_features'] features, substantially reducing
        compute time. If None, all features are used (original behaviour).
    """
    print(f"\n{'='*80}")
    print("SUBGROUP ANALYSIS")
    print(f"{'='*80}")

    if save_dir is None:
        save_dir = os.path.join(OUTPUT_DIR, 'subgroup_analysis')
    os.makedirs(save_dir, exist_ok=True)

    # Determine which features to use for subgroup PI
    top_n_feat = cfg.get('pi_top_n_features', len(feature_names))
    if global_pi_df is not None and top_n_feat < len(feature_names):
        global_top_feats = global_pi_df['feature'].tolist()[:top_n_feat]
        global_top_feats = [f for f in global_top_feats if f in feature_names]
        saving_pct = (1 - len(global_top_feats) / len(feature_names)) * 100
        print(f"\n  ★ Subgroup PI feature filter: global top-{top_n_feat} → "
              f"{len(global_top_feats)} features  (~{saving_pct:.0f}% time saved)")
    else:
        global_top_feats = feature_names
        print(f"\n  Subgroup PI using all {len(feature_names)} features")

    subgroups = _define_subgroups(y_test, y_pred, y_prob)
    print("\n  Subgroup sizes:")
    for name, mask in subgroups.items():
        print(f"    {name:20s}: {mask.sum():6,}  ({mask.mean()*100:5.1f}%)")

    all_pi_rows  = []
    summary_rows = []

    for sg_name, mask in subgroups.items():
        n_sg = mask.sum()
        if n_sg < cfg['min_size']:
            print(f"\n  Skipping {sg_name}: only {n_sg} samples")
            continue

        print(f"\n{'─'*70}")
        print(f"  Subgroup: {sg_name}  (n={n_sg:,})")
        sg_dir = os.path.join(save_dir, sg_name)
        os.makedirs(sg_dir, exist_ok=True)

        X_sg    = X_test[mask].reset_index(drop=True)
        y_sg    = y_test[mask]
        yp_sg   = y_pred[mask]
        prob_sg = y_prob[mask]

        has_both = len(np.unique(y_sg)) == 2

        # Features available in this subgroup after applying global top-N filter
        feats_for_pi = [f for f in global_top_feats if f in X_sg.columns]
        print(f"    PI feature count: {len(feats_for_pi)}")

        # ------------------------------------------------------------------ #
        # 1. PI — skip if CSV checkpoint exists                               #
        # ------------------------------------------------------------------ #
        pi_csv = os.path.join(sg_dir, 'pi_importance.csv')
        if os.path.exists(pi_csv):
            print(f"  [1/3] ✅ PI checkpoint found, loading: {pi_csv}")
            pi_sg = pd.read_csv(pi_csv)
            top_feat_names_sg = pi_sg['feature'].tolist()[:cfg['local_n_top_feats']]
        elif has_both:
            print(f"  [1/3] Permutation Importance ({len(feats_for_pi)} features)...")
            t0 = time.time()
            pi_sg = compute_permutation_importance(
                model, X_sg, y_sg, feats_for_pi,   # filtered feature list
                n_repeats=cfg['pi_n_repeats'],
                subset_size=min(cfg['pi_subset_size'], n_sg))
            pi_sg['subgroup'] = sg_name
            pi_sg.to_csv(pi_csv, index=False)
            all_pi_rows.append(pi_sg)
            print(f"    PI done in {(time.time()-t0)/60:.1f} min")

            top_n_sg  = min(cfg['top_n_display'], len(pi_sg))
            top_pi_sg = pi_sg.head(top_n_sg)
            fig, ax = plt.subplots(figsize=(10, max(5, top_n_sg * 0.38)))
            ax.barh(range(top_n_sg), top_pi_sg['mean_importance'],
                    xerr=top_pi_sg['std_importance'],
                    color='coral', ecolor='gray', capsize=3, alpha=0.85)
            ax.set_yticks(range(top_n_sg))
            ax.set_yticklabels(top_pi_sg['feature'], fontsize=8)
            ax.invert_yaxis()
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xlabel('Mean AUC drop', fontsize=11)
            ax.set_title(
                f'{sg_name} — PI\nn={n_sg:,}  '
                f'(top-{len(feats_for_pi)} features from global PI)',
                fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(sg_dir, 'pi_bar.png'), dpi=300, bbox_inches='tight')
            plt.close()
            top_feat_names_sg = pi_sg['feature'].tolist()[:cfg['local_n_top_feats']]
        else:
            print(f"  [1/3] PI skipped — single-class subgroup")
            pi_sg = None
            top_feat_names_sg = feats_for_pi[:cfg['local_n_top_feats']]

        # PI ordering for Beeswarm: use subgroup order when available, else global order
        pi_order_sg  = pi_sg['feature'].tolist() if pi_sg is not None else feats_for_pi[:]
        feat_idx_map = {f: i for i, f in enumerate(feature_names)}

        # ------------------------------------------------------------------ #
        # 2. Beeswarm — resume from batch-level checkpoints                   #
        # ------------------------------------------------------------------ #
        bee_png     = os.path.join(sg_dir, 'beeswarm.png')
        bee_npy     = os.path.join(sg_dir, 'beeswarm_shap_values.npy')
        ckpt_dir_sg = os.path.join(sg_dir, 'shap_checkpoints')

        if os.path.exists(bee_png) and os.path.exists(bee_npy):
            print(f"  [2/3] ✅ Beeswarm already exists, skipping")
        else:
            print(f"  [2/3] Beeswarm SHAP...")
            n_bg_sg   = min(cfg['beeswarm_n_bg'], n_sg // 2, 30)
            bg_idx_sg = np.random.default_rng(1).choice(n_sg, n_bg_sg, replace=False)
            X_bg_sg   = X_sg.iloc[bg_idx_sg].reset_index(drop=True)

            X_exp_sg, _ = _quartile_label_sample(
                X_sg, y_sg, pi_feature_order=pi_order_sg,
                feature_names=feature_names,
                top_k=cfg['beeswarm_top_k'],
                n_quartiles=cfg['beeswarm_n_quartiles'],
                n_per_cell=cfg['beeswarm_n_per_cell'], random_state=2)
            if len(X_exp_sg) > cfg['beeswarm_n']:
                cap_idx  = np.random.default_rng(3).choice(
                    len(X_exp_sg), cfg['beeswarm_n'], replace=False)
                X_exp_sg = X_exp_sg.iloc[cap_idx].reset_index(drop=True)

            t0 = time.time()
            shap_sg = compute_shap_small_with_checkpoint(
                model, X_exp_sg, X_bg_sg, feature_names,
                checkpoint_dir=ckpt_dir_sg, prefix=f'{sg_name}_bee',
                max_evals=cfg['beeswarm_max_evals'], batch_size=50)
            np.save(bee_npy, shap_sg)
            print(f"    Beeswarm done in {(time.time()-t0)/60:.1f} min")

            top_n_sg         = min(cfg['top_n_display'], len(pi_order_sg))
            ordered_i        = [feat_idx_map[f] for f in pi_order_sg[:top_n_sg]
                                 if f in feat_idx_map]
            ordered_names_sg = [feature_names[i] for i in ordered_i]
            shap_bee_ord     = shap_sg[:, ordered_i]
            X_bee_ord        = X_exp_sg[ordered_names_sg]

            top_bee = min(cfg['top_n_display'], len(ordered_names_sg))
            fig, ax = plt.subplots(figsize=(12, max(5, top_bee * 0.45)))
            plt.sca(ax)
            shap.summary_plot(
                shap_bee_ord, X_bee_ord,
                feature_names=ordered_names_sg,
                max_display=top_bee, plot_type='dot',
                show=False)
            ax = plt.gca()
            ax.set_title(
                f'{sg_name} — Beeswarm  '
                f'(n={len(X_exp_sg)}, quartile×label stratified)',
                fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(bee_png, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    Beeswarm saved.")

        # ------------------------------------------------------------------ #
        # 3. Local SHAP — npy checkpoint                                      #
        # ------------------------------------------------------------------ #
        print(f"  [3/3] Local SHAP...")
        local_sg_dir  = os.path.join(sg_dir, 'local_shap')
        os.makedirs(local_sg_dir, exist_ok=True)

        local_npy     = os.path.join(local_sg_dir, 'local_shap_values.npy')
        local_idx_npy = os.path.join(local_sg_dir, 'local_shap_indices.npy')
        local_ckpt    = os.path.join(local_sg_dir, 'shap_checkpoints')

        local_idx_sg = _collect_local_indices(
            y_sg, yp_sg, prob_sg, X_sg, top_feat_names_sg,
            cfg={'n_per_outcome':    cfg['local_n_per_label'],
                 'n_per_feat_value': cfg['local_n_per_feat'],
                 'n_top_features':   cfg['local_n_top_feats']})

        if len(local_idx_sg) == 0:
            print(f"    No samples selected — skipping")
            summary_rows.append({'subgroup': sg_name, 'size': n_sg,
                                 'local_shap': 'skipped'})
            continue

        X_local_sg  = X_sg.iloc[local_idx_sg].reset_index(drop=True)
        y_local_sg  = y_sg[local_idx_sg]
        yp_local_sg = yp_sg[local_idx_sg]
        pr_local_sg = prob_sg[local_idx_sg]

        if (os.path.exists(local_npy) and os.path.exists(local_idx_npy) and
                np.load(local_idx_npy).tolist() == local_idx_sg):
            print(f"    ✅ Local SHAP checkpoint found, loading")
            shap_local_sg = np.load(local_npy)
        else:
            bg_idx_l = np.random.default_rng(3).choice(
                n_sg, min(cfg['local_n_background'], n_sg), replace=False)
            X_bg_l = X_sg.iloc[bg_idx_l].reset_index(drop=True)
            t0 = time.time()
            shap_local_sg = compute_shap_small_with_checkpoint(
                model, X_local_sg, X_bg_l, feature_names,
                checkpoint_dir=local_ckpt, prefix=f'{sg_name}_local',
                max_evals=cfg['local_max_evals'], batch_size=50)
            np.save(local_npy, shap_local_sg)
            np.save(local_idx_npy, np.array(local_idx_sg))
            print(f"    Local SHAP done in {(time.time()-t0)/60:.1f} min")

        local_exps = []
        for i, sg_i in enumerate(local_idx_sg):
            sv       = shap_local_sg[i]
            fsorted  = sorted(zip(feature_names, sv), key=lambda x: x[1], reverse=True)
            y_pred_i = int(yp_local_sg[i])
            y_true_i = int(y_local_sg[i])
            outcome  = {(1, 1): 'TP', (1, 0): 'FP',
                        (0, 0): 'TN', (0, 1): 'FN'}.get((y_pred_i, y_true_i), '?')
            local_exps.append({
                'subgroup_index': int(sg_i), 'outcome': outcome,
                'prediction':     y_pred_i, 'probability': float(pr_local_sg[i]),
                'true_label':     y_true_i,
                'top_positive':   {f: float(s) for f, s in fsorted[:5]},
                'top_negative':   {f: float(s) for f, s in fsorted[-5:]},
            })

            wf_path = os.path.join(local_sg_dir, f'waterfall_{outcome}_sg{sg_i}.png')
            if not os.path.exists(wf_path):
                fig, _ = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=sv,
                        base_values=float(prob_sg.mean()),
                        data=X_local_sg.iloc[i].values,
                        feature_names=feature_names),
                    max_display=LOCAL_SHAP_CONFIG['max_waterfall'], show=False)
                plt.title(f'{sg_name} | {outcome}  prob={pr_local_sg[i]:.3f}',
                          fontsize=10)
                plt.tight_layout()
                plt.savefig(wf_path, dpi=300, bbox_inches='tight')
                plt.close()

        with open(os.path.join(local_sg_dir, 'local_explanations.json'), 'w') as fh:
            json.dump(local_exps, fh, indent=2, ensure_ascii=False)
        print(f"    Waterfall plots: {len(local_exps)}")

        top1 = top_feat_names_sg[0] if top_feat_names_sg else 'N/A'
        summary_rows.append({'subgroup':            sg_name,
                              'size':               int(n_sg),
                              'top_feature':        top1,
                              'local_shap_samples': len(local_idx_sg)})

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(save_dir, 'subgroup_summary.csv'), index=False)

    print(f"\n  ✓ Subgroup analysis complete")
    return summary_rows


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_tabpfn_pipeline(
    fold: int = 1,
    run_global:   bool = True,
    run_local:    bool = True,
    run_subgroup: bool = True,
):
    print(f"\n{'='*80}")
    print("TabPFN Pipeline V4-resume-top50")
    print(f"{'='*80}")
    t_pipeline = time.time()

    for sub in ['predictions', 'global_importance', 'local_shap', 'subgroup_analysis']:
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

    # ---- 1. Load ----
    print(f"\n{'='*80}")
    print("Step 1: Load data")
    print(f"{'='*80}")
    X_train, y_train, X_test, y_test, feature_names = load_and_preprocess(fold)

    # Detect checkpoint state for time estimation
    global_pi_path = os.path.join(OUTPUT_DIR, 'global_importance',
                                  'pi_feature_importance.csv')
    global_pi_done   = os.path.exists(global_pi_path)
    ckpt_dir_bee     = os.path.join(OUTPUT_DIR, 'global_importance', 'shap_checkpoints')
    bee_done_batches = 0
    if os.path.exists(ckpt_dir_bee):
        files = [f for f in os.listdir(ckpt_dir_bee)
                 if f.startswith('global_bee_batch_')]
        bee_done_batches = len(files)
    bee_samples = (GLOBAL_BEESWARM_CONFIG['top_k_features'] *
                   GLOBAL_BEESWARM_CONFIG['n_quartiles'] * 2 *
                   GLOBAL_BEESWARM_CONFIG['n_per_cell'])
    bee_total = (bee_samples + GLOBAL_BEESWARM_CONFIG['batch_size'] - 1) \
                // GLOBAL_BEESWARM_CONFIG['batch_size']

    estimate_remaining_time(
        feature_names,
        n_subgroups=11,
        global_pi_done=global_pi_done,
        global_bee_batches_done=bee_done_batches,
        global_bee_total_batches=bee_total)

    # ---- 2. Train ----
    print(f"\n{'='*80}")
    print(f"Step 2: Train TabPFN ensemble  ({len(feature_names)} features)")
    print(f"{'='*80}")
    pred_csv = os.path.join(OUTPUT_DIR, 'predictions', f'predictions_fold_{fold}.csv')
    t0 = time.time()
    ensemble = TabPFNEnsemble(
        ENSEMBLE_CONFIG, TABPFN_PARAMS,
        max_samples=TABPFN_MAX_SAMPLES,
        max_features=TABPFN_MAX_FEATURES)
    ensemble.fit(X_train, y_train, feature_names)
    print(f"  ✓ {(time.time()-t0)/60:.1f} min")

    # ---- 3. Predict ----
    print(f"\n{'='*80}")
    print("Step 3: Predictions")
    print(f"{'='*80}")
    if os.path.exists(pred_csv):
        print(f"  ✅ Predictions already exist, loading")
        pred_df = pd.read_csv(pred_csv)
        y_prob  = pred_df['prob_1'].values
        y_pred  = pred_df['prediction'].values
    else:
        t0 = time.time()
        y_proba = ensemble.predict_proba(X_test, feature_names,
                                         batch_size=PREDICT_BATCH_SIZE)
        y_prob  = y_proba[:, 1]
        y_pred  = (y_prob > 0.5).astype(int)
        pred_df = pd.DataFrame({
            'prob_0':     y_proba[:, 0], 'prob_1': y_prob,
            'prediction': y_pred,        'true_label': y_test,
        })
        pred_df.to_csv(pred_csv, index=False)
        print(f"  ✓ {(time.time()-t0):.1f}s")

    metrics_path = os.path.join(OUTPUT_DIR, 'predictions',
                                f'metrics_fold_{fold}.json')
    if not os.path.exists(metrics_path):
        metrics = compute_detailed_metrics(y_test, y_pred, y_prob)
        with open(metrics_path, 'w') as fh:
            json.dump(metrics, fh, indent=2)
    else:
        with open(metrics_path) as fh:
            metrics = json.load(fh)

    print(f"  ROC-AUC={metrics['roc_auc']:.4f}  F1={metrics['macro_f1']:.4f}")

    best_idx   = int(np.argmax(ensemble.oob_scores))
    best_model = ensemble.models[best_idx]['model']
    print(f"  Best model: bag {best_idx+1}  OOB={ensemble.oob_scores[best_idx]:.4f}")

    xai_results = {}

    # ---- 4. Global ----
    if run_global:
        print(f"\n{'='*80}")
        print("Step 4: Global Feature Analysis")
        print(f"{'='*80}")
        t0 = time.time()
        pi_df, _, _ = global_feature_analysis(
            best_model, X_test, y_test, feature_names,
            save_dir=os.path.join(OUTPUT_DIR, 'global_importance'))
        xai_results['global_pi'] = pi_df
        print(f"  ✓ {(time.time()-t0)/60:.1f} min")
    else:
        # Even when skipping global analysis, load existing PI for subgroup filtering
        if os.path.exists(global_pi_path):
            pi_df = pd.read_csv(global_pi_path)
            print(f"  [Global PI loaded from checkpoint for subgroup filtering]")
        else:
            pi_df = pd.DataFrame({'feature':         feature_names,
                                  'mean_importance': 0,
                                  'rank':            range(1, len(feature_names)+1)})

    top_feat_names = pi_df['feature'].tolist()[:LOCAL_SHAP_CONFIG['n_top_features']]

    # ---- 5. Local SHAP ----
    if run_local:
        print(f"\n{'='*80}")
        print("Step 5: Local SHAP Analysis")
        print(f"{'='*80}")
        t0 = time.time()
        local_exp = local_shap_analysis(
            best_model, X_test, y_test, y_pred, y_prob, feature_names,
            top_feature_names=top_feat_names,
            save_dir=os.path.join(OUTPUT_DIR, 'local_shap'))
        xai_results['local'] = local_exp
        print(f"  ✓ {(time.time()-t0)/60:.1f} min")

    # ---- 6. Subgroup — pass global_pi_df to enable top-N feature filtering ----
    if run_subgroup:
        print(f"\n{'='*80}")
        print("Step 6: Subgroup Analysis")
        print(f"{'='*80}")
        t0 = time.time()
        sg_summary = subgroup_analysis(
            best_model, X_test, y_test, y_pred, y_prob, feature_names,
            save_dir=os.path.join(OUTPUT_DIR, 'subgroup_analysis'),
            global_pi_df=pi_df)           # enables top-N feature filtering
        xai_results['subgroup'] = sg_summary
        print(f"  ✓ {(time.time()-t0)/60:.1f} min")

    total = time.time() - t_pipeline
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETE — {total/3600:.2f} h  ({total/60:.0f} min)")
    print(f"Results → {OUTPUT_DIR}")
    print(f"{'='*80}")
    return pred_df, metrics, xai_results


# ===========================================================================
# ENTRY POINT
# ===========================================================================
if __name__ == "__main__":
    if TABPFN_PARAMS.get('device') == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        TABPFN_PARAMS['device'] = 'cpu'
    elif torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    pred_df, metrics, results = run_tabpfn_pipeline(
        fold=1,
        run_global=False,
        run_local=False,
        run_subgroup=True,
    )
