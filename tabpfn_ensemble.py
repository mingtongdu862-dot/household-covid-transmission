"""
TabPFN Bagging Ensemble

Defines the core classes for training and running inference with a
bagging ensemble of TabPFN v2 classifiers.  This module is designed to
be imported by training and explainability scripts; it does not execute
any code when run directly.

Classes
-------
SamplingStrategy
    Static methods for partitioning training data into bags:
    stratified-random, bootstrap, and diversity-maximising splits.

FeatureSelector
    Static methods for partitioning features into groups when the total
    number of features exceeds the TabPFN limit (2,000).

TabPFNEnsemble
    Fits K base TabPFN classifiers on independently drawn bags,
    optionally balances class distributions within each bag, evaluates
    out-of-bag (OOB) AUC for model weighting, and aggregates predictions
    via soft voting, weighted voting, or median.
"""

import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabpfn import TabPFNClassifier


# ===========================================================================
# SAMPLING STRATEGY
# ===========================================================================

class SamplingStrategy:
    """Static factory methods that partition training data into bags."""

    @staticmethod
    def stratified_random(
        X, y, n_bags: int, bag_size: int,
        overlap: float = 0.0, random_state: int = 42,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Stratified random sampling with optional bag overlap.

        When overlap=0.0 the data are partitioned into non-overlapping folds
        via StratifiedKFold; each fold becomes one bag.  When overlap > 0
        each bag draws independently from the full dataset.

        Args:
            X:            Feature array or DataFrame.
            y:            Label array.
            n_bags:       Number of bags to create.
            bag_size:     Target number of training samples per bag.
            overlap:      Fraction of shared samples between bags
                          (0.0 = no overlap).
            random_state: NumPy random seed.

        Returns:
            List of (train_indices, oob_indices) tuples, one per bag.
        """
        np.random.seed(random_state)
        n_total = len(X)
        bags    = []

        if overlap == 0.0:
            indices = np.arange(n_total)
            np.random.shuffle(indices)

            skf = StratifiedKFold(n_splits=n_bags, shuffle=True,
                                  random_state=random_state)
            for _, bag_idx in skf.split(X, y):
                if len(bag_idx) > bag_size:
                    bag_idx = np.random.choice(bag_idx, bag_size, replace=False)
                oob_idx = np.setdiff1d(indices, bag_idx)
                bags.append((bag_idx, oob_idx))
        else:
            all_indices = np.arange(n_total)
            for i in range(n_bags):
                bag_idx, oob_idx = train_test_split(
                    all_indices,
                    train_size=min(bag_size, n_total),
                    stratify=y,
                    random_state=random_state + i,
                )
                bags.append((bag_idx, oob_idx))

        return bags

    @staticmethod
    def bootstrap(
        X, y, n_bags: int, bag_size: int, random_state: int = 42,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Bootstrap sampling (with replacement).

        Each bag draws bag_size samples with replacement; the OOB set
        contains all samples not drawn into the bag.

        Args:
            X:            Feature array or DataFrame.
            y:            Label array.
            n_bags:       Number of bags to create.
            bag_size:     Target number of training samples per bag.
            random_state: NumPy random seed.

        Returns:
            List of (train_indices, oob_indices) tuples, one per bag.
        """
        np.random.seed(random_state)
        n_total = len(X)
        bags    = []

        for i in range(n_bags):
            np.random.seed(random_state + i)
            bag_idx = np.random.choice(n_total, min(bag_size, n_total), replace=True)
            oob_idx = np.setdiff1d(np.arange(n_total), np.unique(bag_idx))
            bags.append((bag_idx, oob_idx))

        return bags

    @staticmethod
    def diversity(
        X, y, n_bags: int, bag_size: int, random_state: int = 42,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Diversity-maximising sampling via non-overlapping stratified partitions.

        The training data are divided into n_bags nearly equal stratified
        partitions; each partition becomes a separate bag.  This maximises
        the diversity of training distributions across base models.

        Args:
            X:            Feature array or DataFrame.
            y:            Label array.
            n_bags:       Number of bags to create.
            bag_size:     Maximum number of training samples per bag.
            random_state: NumPy random seed.

        Returns:
            List of (train_indices, oob_indices) tuples, one per bag.
        """
        np.random.seed(random_state)
        skf  = StratifiedKFold(n_splits=n_bags, shuffle=True,
                               random_state=random_state)
        bags = []

        for train_idx, test_idx in skf.split(X, y):
            if len(train_idx) > bag_size:
                train_idx = np.random.choice(train_idx, bag_size, replace=False)
            bags.append((train_idx, test_idx))

        return bags


# ===========================================================================
# FEATURE SELECTOR
# ===========================================================================

class FeatureSelector:
    """
    Static methods for partitioning features into groups.

    Used when the total number of features exceeds TabPFN's limit of 2,000.
    """

    @staticmethod
    def random_groups(
        feature_names, n_groups: int,
        overlap: float = 0.1, random_state: int = 42,
        max_features_per_group: int = 2_000,
    ) -> List[List[str]]:
        """
        Partition features into n_groups random groups with optional overlap.

        Args:
            feature_names:          Array-like of feature name strings.
            n_groups:               Number of feature groups to create.
            overlap:                Fraction of features shared between groups
                                    (0 = non-overlapping sequential windows).
            random_state:           NumPy random seed.
            max_features_per_group: Maximum features per group.

        Returns:
            List of feature-name lists, one per group.
        """
        np.random.seed(random_state)
        n_features = len(feature_names)
        groups     = []

        for i in range(n_groups):
            if overlap == 0:
                start          = i * max_features_per_group
                end            = min((i + 1) * max_features_per_group, n_features)
                group_features = feature_names[start:end]
            else:
                group_features = np.random.choice(
                    feature_names,
                    size=min(max_features_per_group, n_features),
                    replace=False,
                )
            groups.append(list(group_features))

        return groups

    @staticmethod
    def all_features(feature_names) -> List[List[str]]:
        """
        Return all features as a single group (valid when n_features ≤ limit).

        Args:
            feature_names: Array-like of feature name strings.

        Returns:
            Single-element list containing all feature names.
        """
        return [feature_names]


# ===========================================================================
# TABPFN ENSEMBLE
# ===========================================================================

class TabPFNEnsemble:
    """
    Bagging ensemble of TabPFN v2 classifiers.

    Supports stratified-random, bootstrap, and diversity bagging strategies;
    optional within-bag class balancing (undersample / oversample / combined);
    OOB-AUC-based model weighting; and soft-voting, weighted-voting, or
    median aggregation at prediction time.

    Parameters
    ----------
    config : dict
        Ensemble configuration dictionary (see tabpfn_config.ENSEMBLE_CONFIG).
    tabpfn_params : dict
        Keyword arguments forwarded to TabPFNClassifier (see
        tabpfn_config.TABPFN_PARAMS).
    max_samples : int
        Hard upper limit on samples per bag imposed by the TabPFN model.
    max_features : int
        Hard upper limit on features per base model imposed by TabPFN.
    """

    def __init__(
        self,
        config: dict,
        tabpfn_params: dict,
        max_samples: int = 50_000,
        max_features: int = 2_000,
    ):
        self.config        = config
        self.tabpfn_params = tabpfn_params
        self.max_samples   = max_samples
        self.max_features  = max_features

        self.models        = []
        self.bag_indices   = []
        self.feature_groups = []
        self.oob_scores    = []
        self.weights       = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _balance_data(
        self,
        X, y,
        indices: np.ndarray,
        strategy: str = 'undersample',
        target_ratio: float = 0.5,
        random_state: int = 42,
    ) -> np.ndarray:
        """
        Re-sample indices to achieve the target minority-class ratio.

        Args:
            X:            Full feature array (used only to obtain shape).
            y:            Full label array.
            indices:      Subset of row indices to re-balance.
            strategy:     'undersample' | 'oversample' | 'combined'.
            target_ratio: Desired fraction of minority-class samples.
            random_state: NumPy random seed.

        Returns:
            np.ndarray: Re-balanced index array (shuffled).
        """
        np.random.seed(random_state)
        y_subset = y[indices]

        unique, counts = np.unique(y_subset, return_counts=True)
        if len(unique) < 2:
            return indices  # Only one class present; return unchanged

        majority_class   = unique[np.argmax(counts)]
        minority_class   = unique[np.argmin(counts)]
        majority_indices = indices[y_subset == majority_class]
        minority_indices = indices[y_subset == minority_class]

        n_minority = len(minority_indices)
        n_majority = len(majority_indices)

        if strategy == 'undersample':
            n_target_majority = int(n_minority / target_ratio - n_minority)
            n_target_majority = min(n_target_majority, n_majority)
            selected_majority = np.random.choice(
                majority_indices, size=n_target_majority, replace=False)
            balanced_indices = np.concatenate([minority_indices, selected_majority])

        elif strategy == 'oversample':
            n_target_minority = int(n_majority * target_ratio / (1 - target_ratio))
            selected_minority = np.random.choice(
                minority_indices, size=n_target_minority,
                replace=(n_target_minority > n_minority))
            balanced_indices = np.concatenate([selected_minority, majority_indices])

        elif strategy == 'combined':
            # Modest reduction of total size with oversampling and undersampling
            n_target_total   = int((n_minority + n_majority) * 0.8)
            n_target_minority = int(n_target_total * target_ratio)
            n_target_majority = n_target_total - n_target_minority

            selected_minority = np.random.choice(
                minority_indices, size=n_target_minority,
                replace=(n_target_minority > n_minority))
            selected_majority = np.random.choice(
                majority_indices,
                size=min(n_target_majority, n_majority),
                replace=False)
            balanced_indices = np.concatenate([selected_minority, selected_majority])

        else:
            raise ValueError(f'Unknown balance strategy: {strategy!r}')

        np.random.shuffle(balanced_indices)
        return balanced_indices

    def _create_feature_groups(self, X, y, feature_names) -> List[List[str]]:
        """Select feature groups based on the configured feature strategy."""
        n_features = len(feature_names)

        if n_features <= self.max_features:
            print(f'  Features within limit ({n_features} ≤ {self.max_features}); '
                  f'using all features')
            return [feature_names]

        strategy = self.config['feature_strategy']
        print(f'  Features exceed limit ({n_features} > {self.max_features})')
        print(f'  Feature strategy: {strategy}')

        if strategy == 'random_groups':
            n_groups = self.config.get('n_feature_groups', 3)
            return FeatureSelector.random_groups(
                feature_names, n_groups,
                overlap=self.config.get('feature_overlap', 0.1),
                random_state=self.config['random_state'],
                max_features_per_group=self.max_features,
            )
        elif strategy == 'all':
            raise ValueError(
                f"Cannot use 'all' feature strategy with {n_features} features "
                f'(limit: {self.max_features})'
            )
        else:
            raise ValueError(f'Unknown feature strategy: {strategy!r}')

    def _create_sample_bags(self, X, y) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Partition training data into bags using the configured bagging strategy."""
        strategy = self.config['bagging_strategy']
        n_bags   = self.config['n_bags']
        bag_size = min(self.config['bag_sample_size'], self.max_samples, len(X))

        print(f'  Creating {n_bags} bags of size {bag_size:,}...')

        if strategy == 'stratified_random':
            bags = SamplingStrategy.stratified_random(
                X, y, n_bags, bag_size,
                overlap=self.config['bag_overlap'],
                random_state=self.config['random_state'],
            )
        elif strategy == 'bootstrap':
            bags = SamplingStrategy.bootstrap(
                X, y, n_bags, bag_size,
                random_state=self.config['random_state'],
            )
        elif strategy == 'diversity':
            bags = SamplingStrategy.diversity(
                X, y, n_bags, bag_size,
                random_state=self.config['random_state'],
            )
        else:
            raise ValueError(f'Unknown bagging strategy: {strategy!r}')

        for i, (train_idx, oob_idx) in enumerate(bags):
            print(f'    Bag {i+1}: {len(train_idx):,} train, {len(oob_idx):,} OOB')

        return bags

    def _compute_weights(self) -> None:
        """
        Compute base-model ensemble weights from OOB AUC scores.

        Uses a softmax-style transform (temperature=5) so that models with
        higher OOB AUC receive exponentially more weight.  Falls back to
        uniform weights when OOB weighting is disabled.
        """
        if not self.config['use_oob_weighting']:
            self.weights = np.ones(len(self.models)) / len(self.models)
            return

        oob_scores = np.nan_to_num(np.array(self.oob_scores), nan=0.5)
        weights    = np.exp(oob_scores * 5)   # Temperature = 5
        self.weights = weights / weights.sum()

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X, y, feature_names: List[str]) -> 'TabPFNEnsemble':
        """
        Train the full bagging ensemble.

        For each (bag, feature-group) combination:
            1. Optionally re-balance the class distribution.
            2. Fit a TabPFNClassifier.
            3. Evaluate OOB AUC for weighting.

        Args:
            X:             Training features (DataFrame or ndarray).
            y:             Training labels (ndarray, shape n_samples).
            feature_names: Ordered list of feature name strings.

        Returns:
            self
        """
        print(f"\n{'='*60}")
        print('Training TabPFN Ensemble')
        print(f"{'='*60}")
        print(f'  Strategy:       {self.config["bagging_strategy"]}')
        print(f'  N bags:         {self.config["n_bags"]}')
        print(f'  Bag size:       {self.config["bag_sample_size"]:,}')
        print(f'  Total samples:  {len(X):,}')
        print(f'  Total features: {len(feature_names):,}')

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)

        self.feature_groups = self._create_feature_groups(X, y, feature_names)
        self.bag_indices    = self._create_sample_bags(X, y)

        n_feature_variants = len(self.feature_groups)
        n_sample_bags      = len(self.bag_indices)
        total_models       = n_sample_bags * n_feature_variants

        print(f'  Total models:   {total_models} '
              f'({n_sample_bags} bags × {n_feature_variants} feature groups)')

        fit_times = []
        model_idx = 0

        for bag_idx, (train_idx, oob_idx) in enumerate(self.bag_indices):
            for group_idx, features in enumerate(self.feature_groups):
                model_idx += 1
                print(f'\n  [{model_idx}/{total_models}] '
                      f'bag {bag_idx+1}/{n_sample_bags}, '
                      f'feature group {group_idx+1}/{n_feature_variants}')

                # Optional class balancing within the bag
                if self.config.get('balance_classes', False):
                    original_size = len(train_idx)
                    train_idx     = self._balance_data(
                        X, y, train_idx,
                        strategy=self.config.get('balance_strategy', 'undersample'),
                        target_ratio=self.config.get('target_ratio', 0.5),
                        random_state=self.config['random_state'] + bag_idx,
                    )
                    pos_ratio = (y[train_idx] == 1).mean()
                    print(f'    Balanced: {original_size:,} → {len(train_idx):,} '
                          f'samples, {pos_ratio:.1%} positive')

                X_train_bag = X.iloc[train_idx][features]
                y_train_bag = y[train_idx]

                start_time = time.time()
                model      = TabPFNClassifier(**self.tabpfn_params)
                model.fit(X_train_bag, y_train_bag)
                fit_time   = time.time() - start_time
                fit_times.append(fit_time)

                print(f'    Fitted in {fit_time:.1f}s '
                      f'({len(train_idx):,} samples, {len(features):,} features)')

                # OOB evaluation for model weighting
                oob_score = None
                if len(oob_idx) > 0 and self.config['use_oob_weighting']:
                    X_oob = X.iloc[oob_idx][features]
                    y_oob = y[oob_idx]

                    if len(oob_idx) > 5_000:
                        sample_idx = np.random.choice(len(oob_idx), 5_000, replace=False)
                        X_oob = X_oob.iloc[sample_idx]
                        y_oob = y_oob[sample_idx]

                    try:
                        oob_proba = model.predict_proba(X_oob)[:, 1]
                        oob_score = roc_auc_score(y_oob, oob_proba)
                        print(f'    OOB AUC: {oob_score:.4f}')
                    except Exception as e:
                        print(f'    OOB scoring failed: {e}')
                        oob_score = 0.5

                self.models.append({
                    'model':      model,
                    'features':   features,
                    'bag_idx':    bag_idx,
                    'group_idx':  group_idx,
                    'oob_score':  oob_score,
                })
                self.oob_scores.append(oob_score if oob_score is not None else 0.5)

                torch.cuda.empty_cache()

        self._compute_weights()

        print(f"\n{'='*60}")
        print('Ensemble Training Complete')
        print(f"{'='*60}")
        print(f'  Total models:   {len(self.models)}')
        print(f'  Avg fit time:   {np.mean(fit_times):.1f}s')
        print(f'  Total fit time: {np.sum(fit_times):.1f}s')
        if self.weights is not None:
            print(f'  Weight range:   [{self.weights.min():.3f}, {self.weights.max():.3f}]')

        return self

    def predict_proba(
        self,
        X,
        feature_names: List[str],
        batch_size: int = 3_000,
    ) -> np.ndarray:
        """
        Produce class-probability estimates from the ensemble.

        Iterates over all base models and aggregates their predictions using
        the configured ensemble method (soft voting / weighted voting / median).
        Large inputs are processed in mini-batches to limit GPU memory usage.

        Args:
            X:             Test features (DataFrame or ndarray).
            feature_names: Ordered list of feature name strings.
            batch_size:    Maximum number of samples per prediction batch.

        Returns:
            np.ndarray: Shape (n_samples, 2) with [P(class=0), P(class=1)].
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)

        n_samples     = len(X)
        n_models      = len(self.models)
        all_predictions = np.zeros((n_samples, n_models))

        for i, model_dict in enumerate(self.models):
            model    = model_dict['model']
            features = model_dict['features']
            X_model  = X[features]

            if len(X_model) <= batch_size:
                proba = model.predict_proba(X_model)[:, 1]
            else:
                proba_batches = []
                for start in range(0, len(X_model), batch_size):
                    end   = min(start + batch_size, len(X_model))
                    proba_batches.append(
                        model.predict_proba(X_model.iloc[start:end])[:, 1]
                    )
                proba = np.concatenate(proba_batches)

            all_predictions[:, i] = proba
            torch.cuda.empty_cache()

        # ── Aggregation ───────────────────────────────────────────────────────
        method = self.config['ensemble_method']

        if method == 'soft_voting':
            final_proba = all_predictions.mean(axis=1)
        elif method == 'weighted_voting':
            final_proba = np.average(all_predictions, axis=1, weights=self.weights)
        elif method == 'median':
            final_proba = np.median(all_predictions, axis=1)
        else:
            raise ValueError(f'Unknown ensemble method: {method!r}')

        return np.column_stack([1 - final_proba, final_proba])
