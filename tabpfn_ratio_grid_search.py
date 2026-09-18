"""
tabpfn_ratio_grid_search.py
============================
Grid search over the bagging ensemble's training-context positive-class
ratio (``target_ratio`` / ``train_pos_ratio``), addressing Reviewer 1
comment #3:

    "The oversampling ratio was selected to maximise recall while
    maintaining an 'acceptable' positive-class F1, but 'acceptable' is
    not defined. This selection must be conducted exclusively on
    validation data using a pre-specified criterion."

Selection criterion (single objective, no free threshold to justify):
the ratio that **maximises positive-class F1 (F1+) on the validation
fold**. Recall+, PPV, specificity, and ROC-AUC are also recorded at every
grid point for transparency and for the supplementary sweep figure, but
none of them enter the selection rule.

The held-out **test fold is never loaded or touched** by this script --
only ``train_fold_N`` and ``val_fold_N`` are used, matching the reviewer's
explicit requirement.

This reuses ``fit_bags`` / ``predict_with_bags`` / ``compute_metrics`` /
``load_fold_data`` from ``tabpfn_train.py`` rather than duplicating the
bagging logic, so the grid search trains under exactly the same procedure
(sampling, TabPFN params, metric definitions) as the main training script.

Usage
-----
    python tabpfn_ratio_grid_search.py

Outputs written to ``<OUTPUT_DIR>/ratio_grid_search/``
--------------------------------------------------------
- ``ratio_grid_search_fold{k}.csv``    one row per ratio, all recorded metrics
- ``ratio_grid_search_fold{k}.json``   same, plus the selected ratio and criterion
- ``ratio_grid_search_fold{k}.png``    F1+/Recall+/PPV/ROC-AUC vs. ratio, with
                                       the selected ratio marked
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from config import *
import tabpfn_train as T


# ===========================================================================
# GRID SEARCH CONFIGURATION
# ===========================================================================
GRID_SEARCH_CONFIG = {
    # Positive-class ratio grid, matching the manuscript's stated
    # {0%, 10%, ..., 100%} sweep. Ratios of exactly 0.0 or 1.0 give a
    # single-class training context; TabPFN cannot fit those; they are
    # skipped automatically (see run below) rather than crashing the sweep.
    'ratios': [round(r, 2) for r in np.arange(0.0, 1.001, 0.1)],

    # Which fold to run the sweep on. The selected ratio is meant to be a
    # single, global hyperparameter applied uniformly across all folds in
    # the main 5-fold run (this is how CV_CONFIG['train_pos_ratio'] is
    # already used in tabpfn_train.py), not re-tuned per fold, so running
    # the (expensive: n_bags fits per ratio) sweep once on a single
    # representative fold -- rather than all 5 -- is deliberate, not a
    # shortcut. Re-run with a different value here (or extend the loop to
    # average across multiple folds) if you want to sanity-check stability
    # across folds before committing to one ratio.
    'fold': 1,

    # None -> reuse tabpfn_train.CV_CONFIG's values (same bagging setup as
    # the real training run). Override here to shrink the sweep for a
    # quick smoke test without touching CV_CONFIG itself.
    'n_bags': None,
    'bag_train_size': None,
}


def run_ratio_grid_search():
    print("\n" + "=" * 80)
    print("TabPFN TRAINING-CONTEXT RATIO GRID SEARCH")
    print("Selection criterion: maximise validation F1+ (single objective)")
    print("=" * 80)

    if T.TABPFN_MODEL_PARAMS['device'] == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        T.TABPFN_MODEL_PARAMS['device'] = 'cpu'

    fold = GRID_SEARCH_CONFIG['fold']
    ratios = GRID_SEARCH_CONFIG['ratios']
    n_bags = GRID_SEARCH_CONFIG['n_bags'] or T.CV_CONFIG['n_bags']
    bag_train_size = GRID_SEARCH_CONFIG['bag_train_size'] or T.CV_CONFIG['bag_train_size']

    print(f"  Fold:            {fold}  (validation only -- test fold is not loaded)")
    print(f"  Ratio grid:      {[f'{r:.0%}' for r in ratios]}")
    print(f"  Bags per ratio:  {n_bags}")
    print(f"  Bag train size:  {bag_train_size:,}")
    print("=" * 80)

    # Only train/val are used. test_df is intentionally never referenced
    # below -- the reviewer's requirement is that this selection never
    # touches the held-out test set.
    train_df, val_df, _test_df_unused, feature_names = T.load_fold_data(fold)
    del _test_df_unused

    output_dir = os.path.join(OUTPUT_DIR, 'ratio_grid_search')
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for ratio in ratios:
        print(f"\n{'#' * 80}\n# ratio = {ratio:.0%}\n{'#' * 80}")

        cfg = dict(T.CV_CONFIG)
        cfg['train_pos_ratio'] = ratio
        cfg['n_bags'] = n_bags
        cfg['bag_train_size'] = bag_train_size

        try:
            models, train_idx_used, total_fit_time = T.fit_bags(train_df, feature_names, cfg)
            val_ens, val_bags, val_proba = T.predict_with_bags(
                models, val_df, feature_names, eval_tag=f'Val (ratio={ratio:.0%})')
        except Exception as e:
            print(f"  SKIPPED ratio={ratio:.0%}: {e}")
            rows.append({'ratio': ratio, 'status': 'failed', 'error': str(e)})
            continue

        rows.append({
            'ratio':            ratio,
            'status':           'ok',
            'val_f1_plus':      val_ens['class_1_f1'],
            'val_recall_plus':  val_ens['class_1_recall'],
            'val_ppv':          val_ens['ppv'],
            'val_specificity':  val_ens['specificity'],
            'val_roc_auc':      val_ens['roc_auc'],
            'val_pr_auc':       val_ens['class_1_pr_auc'],
            'val_mcc':          val_ens['mcc'],
            'total_fit_time':   total_fit_time,
        })
        print(f"  ratio={ratio:.0%}  F1+={val_ens['class_1_f1']:.4f}  "
              f"Recall+={val_ens['class_1_recall']:.4f}  PPV={val_ens['ppv']:.4f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f'ratio_grid_search_fold{fold}.csv')
    df.to_csv(csv_path, index=False)

    ok = df[df['status'] == 'ok'].copy()
    if ok.empty:
        raise RuntimeError("Every ratio in the grid failed to fit -- see per-ratio "
                           "errors above. Nothing to select.")

    best_idx   = ok['val_f1_plus'].idxmax()
    best_row   = ok.loc[best_idx]
    best_ratio = float(best_row['ratio'])

    print(f"\n{'=' * 80}")
    print(f"SELECTED RATIO (max validation F1+): {best_ratio:.0%}")
    print(f"  F1+:          {best_row['val_f1_plus']:.4f}")
    print(f"  Recall+:      {best_row['val_recall_plus']:.4f}")
    print(f"  PPV:          {best_row['val_ppv']:.4f}")
    print(f"  ROC-AUC:      {best_row['val_roc_auc']:.4f}")
    print(f"{'=' * 80}")
    print(f"Update CV_CONFIG['train_pos_ratio'] in tabpfn_train.py and "
          f"ENSEMBLE_CONFIG['target_ratio'] in config.py to {best_ratio} "
          f"before the main 5-fold run.")

    result_summary = {
        'fold':                fold,
        'selection_criterion': 'argmax validation F1+ (class_1_f1), test fold not used',
        'grid':                ratios,
        'n_bags':               n_bags,
        'bag_train_size':       bag_train_size,
        'selected_ratio':      best_ratio,
        'selected_ratio_metrics': best_row.to_dict(),
        'all_ratios':          df.to_dict(orient='records'),
    }
    json_path = os.path.join(output_dir, f'ratio_grid_search_fold{fold}.json')
    with open(json_path, 'w') as f:
        json.dump(result_summary, f, indent=2)

    # ── Sweep figure ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ok['ratio'], ok['val_f1_plus'], 'o-', color='#1f77b4', label='F1+ (selection criterion)')
    ax.plot(ok['ratio'], ok['val_recall_plus'], 's--', color='#d62728', label='Recall+')
    ax.plot(ok['ratio'], ok['val_ppv'], '^--', color='#2ca02c', label='PPV')
    ax.plot(ok['ratio'], ok['val_roc_auc'], 'd--', color='#7f7f7f', label='ROC-AUC')
    ax.axvline(best_ratio, color='black', linestyle=':', linewidth=1.5,
              label=f'Selected: {best_ratio:.0%}')
    ax.set_xlabel('Training-context positive-class ratio')
    ax.set_ylabel('Validation-set metric')
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(f'Fold {fold} — Ratio grid search (validation set only)',
                fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    png_path = os.path.join(output_dir, f'ratio_grid_search_fold{fold}.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nResults saved to {output_dir}:")
    print(f"  • CSV:  {csv_path}")
    print(f"  • JSON: {json_path}")
    print(f"  • Plot: {png_path}")

    return result_summary


if __name__ == "__main__":
    run_ratio_grid_search()
