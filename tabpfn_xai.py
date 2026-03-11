"""
tabpfn_xai.py
=============
Explainability analysis script for the TabPFN ensemble model.

This module re-uses the ``TabPFNEnsemble`` class (identical to
``tabpfn_ensemble.py``) and extends the pipeline with model-agnostic
interpretability methods:

    - **Permutation Importance (PI)**: global feature ranking via AUC-drop
      on the held-out test set (n = 3 000, R = 5 repetitions).
    - **Kernel SHAP**: directional feature-effect estimates on a stratified
      subsample that crosses feature quartiles with class labels, used to
      colour-encode the PI bar chart.
    - **Subgroup analysis**: separate PI and Kernel SHAP analyses for
      moderate-risk (0.3 ≤ ŷ < 0.7) and low-risk (ŷ < 0.3) strata.
    - **Local SHAP waterfall plots**: individual-level prediction decompositions
      for representative moderate- and low-risk households.

All plots and raw SHAP/PI values are saved to the output directory defined in
``config.py`` (``OUTPUT_DIR``).

Configuration is loaded from ``config.py``.

Usage
-----
    python tabpfn_xai.py
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabpfn import TabPFNClassifier
import torch
import time
from typing import List, Tuple


# ===========================================================================
# SAMPLING STRATEGY CLASS
# ===========================================================================
class SamplingStrategy:
    """Base class for sampling strategies"""
    
    @staticmethod
    def stratified_random(X, y, n_bags, bag_size, overlap=0.0, random_state=42):
        """
        Stratified random sampling with optional overlap
        
        Returns: List of (train_indices, oob_indices) tuples
        """
        np.random.seed(random_state)
        n_total = len(X)
        bags = []
        
        if overlap == 0.0:
            # Non-overlapping: partition data
            indices = np.arange(n_total)
            np.random.shuffle(indices)
            
            # Stratified split into n_bags
            skf = StratifiedKFold(n_splits=n_bags, shuffle=True, random_state=random_state)
            
            for i, (_, bag_idx) in enumerate(skf.split(X, y)):
                # Sample bag_size from this fold (if fold is larger)
                if len(bag_idx) > bag_size:
                    bag_idx = np.random.choice(bag_idx, bag_size, replace=False)
                
                oob_idx = np.setdiff1d(indices, bag_idx)
                bags.append((bag_idx, oob_idx))
        else:
            # Overlapping: each bag samples independently
            all_indices = np.arange(n_total)
            
            for i in range(n_bags):
                seed = random_state + i
                bag_idx, oob_idx = train_test_split(
                    all_indices, 
                    train_size=min(bag_size, n_total),
                    stratify=y,
                    random_state=seed
                )
                bags.append((bag_idx, oob_idx))
        
        return bags
    
    @staticmethod
    def bootstrap(X, y, n_bags, bag_size, random_state=42):
        """
        Bootstrap sampling (with replacement)
        
        Returns: List of (train_indices, oob_indices) tuples
        """
        np.random.seed(random_state)
        n_total = len(X)
        bags = []
        
        for i in range(n_bags):
            seed = random_state + i
            np.random.seed(seed)
            
            # Sample with replacement
            bag_idx = np.random.choice(n_total, min(bag_size, n_total), replace=True)
            
            # OOB = samples not in bag
            oob_idx = np.setdiff1d(np.arange(n_total), np.unique(bag_idx))
            
            bags.append((bag_idx, oob_idx))
        
        return bags
    
    @staticmethod
    def diversity(X, y, n_bags, bag_size, random_state=42):
        """
        Diversity-focused sampling: maximize coverage with minimal overlap
        
        Strategy:
        - Divide data into n_bags nearly equal partitions
        - Shuffle within each partition
        - Each bag gets a different partition (maximizes diversity)
        """
        np.random.seed(random_state)
        
        # Create stratified partitions
        skf = StratifiedKFold(n_splits=n_bags, shuffle=True, random_state=random_state)
        
        bags = []
        
        for train_idx, test_idx in skf.split(X, y):
            # This fold becomes one bag
            if len(train_idx) > bag_size:
                # If partition too large, sample
                train_idx = np.random.choice(train_idx, bag_size, replace=False)
            
            # OOB = everything not in this partition
            oob_idx = test_idx
            
            bags.append((train_idx, oob_idx))
        
        return bags


# ===========================================================================
# FEATURE SELECTION STRATEGY CLASS
# ===========================================================================
class FeatureSelector:
    """Feature selection strategies for high-dimensional data"""
    
    @staticmethod
    def random_groups(feature_names, n_groups, overlap=0.1, random_state=42, 
                     max_features_per_group=2000):
        """
        Randomly partition features into groups with optional overlap
        """
        np.random.seed(random_state)
        n_features = len(feature_names)
        
        groups = []
        for i in range(n_groups):
            # Sample features
            if overlap == 0:
                start = i * max_features_per_group
                end = min((i + 1) * max_features_per_group, n_features)
                group_features = feature_names[start:end]
            else:
                group_features = np.random.choice(
                    feature_names, 
                    size=min(max_features_per_group, n_features),
                    replace=False
                )
            
            groups.append(list(group_features))
        
        return groups
    
    @staticmethod
    def all_features(feature_names):
        """Return all features as a single group (if within limit)"""
        return [feature_names]


# ===========================================================================
# TABPFN ENSEMBLE CLASS
# ===========================================================================
class TabPFNEnsemble:
    """
    Ensemble of TabPFN models with intelligent sampling and feature selection
    """
    
    def __init__(self, config, tabpfn_params, max_samples=50000, max_features=2000):
        self.config = config
        self.tabpfn_params = tabpfn_params
        self.max_samples = max_samples
        self.max_features = max_features
        self.models = []
        self.bag_indices = []
        self.feature_groups = []
        self.oob_scores = []
        self.weights = None
    
    def _balance_data(self, X, y, indices, strategy='undersample', target_ratio=0.5, random_state=42):
        """
        Balance class distribution in the given indices.
        
        Args:
            X: Full dataset features
            y: Full dataset labels
            indices: Indices to balance
            strategy: 'undersample', 'oversample', or 'combined'
            target_ratio: Target ratio for minority class (0.5 = 50/50)
            random_state: Random seed
            
        Returns:
            Balanced indices
        """
        np.random.seed(random_state)
        
        # Get labels for these indices
        y_subset = y[indices]
        
        # Find majority and minority classes
        unique, counts = np.unique(y_subset, return_counts=True)
        if len(unique) < 2:
            # Only one class present, return as is
            return indices
        
        majority_class = unique[np.argmax(counts)]
        minority_class = unique[np.argmin(counts)]
        
        majority_indices = indices[y_subset == majority_class]
        minority_indices = indices[y_subset == minority_class]
        
        n_minority = len(minority_indices)
        n_majority = len(majority_indices)
        
        if strategy == 'undersample':
            # Undersample majority class to match minority
            n_target_majority = int(n_minority / target_ratio - n_minority)
            n_target_majority = min(n_target_majority, n_majority)
            
            selected_majority = np.random.choice(
                majority_indices, 
                size=n_target_majority, 
                replace=False
            )
            balanced_indices = np.concatenate([minority_indices, selected_majority])
            
        elif strategy == 'oversample':
            # Oversample minority class to match majority
            n_target_minority = int(n_majority * target_ratio / (1 - target_ratio))
            
            # Sample with replacement if needed
            selected_minority = np.random.choice(
                minority_indices,
                size=n_target_minority,
                replace=(n_target_minority > n_minority)
            )
            balanced_indices = np.concatenate([selected_minority, majority_indices])
            
        elif strategy == 'combined':
            # Compromise: oversample minority + undersample majority
            n_target_total = int((n_minority + n_majority) * 0.8)  # Reduce total a bit
            n_target_minority = int(n_target_total * target_ratio)
            n_target_majority = n_target_total - n_target_minority
            
            # Oversample minority if needed
            selected_minority = np.random.choice(
                minority_indices,
                size=n_target_minority,
                replace=(n_target_minority > n_minority)
            )
            
            # Undersample majority if needed
            selected_majority = np.random.choice(
                majority_indices,
                size=min(n_target_majority, n_majority),
                replace=False
            )
            
            balanced_indices = np.concatenate([selected_minority, selected_majority])
        
        else:
            raise ValueError(f"Unknown balance strategy: {strategy}")
        
        # Shuffle
        np.random.shuffle(balanced_indices)
        
        return balanced_indices
        
    def fit(self, X, y, feature_names):
        """
        Train ensemble of TabPFN models
        
        Args:
            X: Training features (DataFrame or array)
            y: Training labels
            feature_names: List of feature names
        """
        print(f"\n{'='*60}")
        print(f"Training TabPFN Ensemble")
        print(f"{'='*60}")
        print(f"  Strategy:        {self.config['bagging_strategy']}")
        print(f"  N bags:          {self.config['n_bags']}")
        print(f"  Bag size:        {self.config['bag_sample_size']:,}")
        print(f"  Total samples:   {len(X):,}")
        print(f"  Total features:  {len(feature_names):,}")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)
        
        # Step 1: Generate feature groups (if needed)
        self.feature_groups = self._create_feature_groups(X, y, feature_names)
        n_feature_variants = len(self.feature_groups)
        
        # Step 2: Generate sample bags
        self.bag_indices = self._create_sample_bags(X, y)
        n_sample_bags = len(self.bag_indices)
        
        total_models = n_sample_bags * n_feature_variants
        print(f"  Total models:    {total_models} ({n_sample_bags} sample bags × {n_feature_variants} feature groups)")
        
        # Step 3: Train all models
        model_idx = 0
        fit_times = []
        
        for bag_idx, (train_idx, oob_idx) in enumerate(self.bag_indices):
            for group_idx, features in enumerate(self.feature_groups):
                model_idx += 1
                print(f"\n  [{model_idx}/{total_models}] Training model (bag {bag_idx+1}/{n_sample_bags}, features {group_idx+1}/{n_feature_variants})...")
                
                # Apply class balancing if enabled
                if self.config.get('balance_classes', False):
                    original_size = len(train_idx)
                    train_idx = self._balance_data(
                        X, y, train_idx,
                        strategy=self.config.get('balance_strategy', 'undersample'),
                        target_ratio=self.config.get('target_ratio', 0.5),
                        random_state=self.config['random_state'] + bag_idx
                    )
                    
                    # Print class distribution after balancing
                    y_train_balanced = y[train_idx]
                    pos_ratio = (y_train_balanced == 1).mean()
                    print(f"      Balanced: {original_size:,} → {len(train_idx):,} samples, {pos_ratio:.1%} positive")
                
                # Prepare data
                X_train_bag = X.iloc[train_idx][features]
                y_train_bag = y[train_idx]
                
                # Train model
                start_time = time.time()
                model = TabPFNClassifier(**self.tabpfn_params)
                model.fit(X_train_bag, y_train_bag)
                fit_time = time.time() - start_time
                fit_times.append(fit_time)
                
                print(f"      Fitted in {fit_time:.1f}s ({len(train_idx):,} samples, {len(features):,} features)")
                
                # Compute OOB score if available
                oob_score = None
                if len(oob_idx) > 0 and self.config['use_oob_weighting']:
                    X_oob = X.iloc[oob_idx][features]
                    y_oob = y[oob_idx]
                    
                    # Sample OOB if too large
                    if len(oob_idx) > 5000:
                        sample_idx = np.random.choice(len(oob_idx), 5000, replace=False)
                        X_oob = X_oob.iloc[sample_idx]
                        y_oob = y_oob[sample_idx]
                    
                    try:
                        oob_proba = model.predict_proba(X_oob)[:, 1]
                        oob_score = roc_auc_score(y_oob, oob_proba)
                        print(f"      OOB AUC: {oob_score:.4f}")
                    except Exception as e:
                        print(f"      OOB scoring failed: {e}")
                        oob_score = 0.5
                
                self.models.append({
                    'model': model,
                    'features': features,
                    'bag_idx': bag_idx,
                    'group_idx': group_idx,
                    'oob_score': oob_score
                })
                self.oob_scores.append(oob_score if oob_score else 0.5)
                
                # Clean up
                torch.cuda.empty_cache()
        
        # Step 4: Compute ensemble weights
        self._compute_weights()
        
        print(f"\n{'='*60}")
        print(f"Ensemble Training Complete")
        print(f"{'='*60}")
        print(f"  Total models:    {len(self.models)}")
        print(f"  Avg fit time:    {np.mean(fit_times):.1f}s")
        print(f"  Total fit time:  {np.sum(fit_times):.1f}s")
        if self.weights is not None:
            print(f"  Weight range:    [{self.weights.min():.3f}, {self.weights.max():.3f}]")
        
        return self
    
    def predict_proba(self, X, feature_names, batch_size=3000):
        """
        Ensemble prediction with batching
        
        Args:
            X: Test features
            feature_names: List of feature names
            batch_size: Batch size for prediction
            
        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=feature_names)
        
        n_samples = len(X)
        n_models = len(self.models)
        
        # Collect predictions from all models
        all_predictions = np.zeros((n_samples, n_models))
        
        for i, model_dict in enumerate(self.models):
            model = model_dict['model']
            features = model_dict['features']
            
            # Batch prediction
            X_model = X[features]
            
            if len(X_model) <= batch_size:
                proba = model.predict_proba(X_model)[:, 1]
            else:
                # Batched prediction
                proba_batches = []
                for start in range(0, len(X_model), batch_size):
                    end = min(start + batch_size, len(X_model))
                    batch_proba = model.predict_proba(X_model.iloc[start:end])[:, 1]
                    proba_batches.append(batch_proba)
                proba = np.concatenate(proba_batches)
            
            all_predictions[:, i] = proba
            
            torch.cuda.empty_cache()
        
        # Ensemble aggregation
        if self.config['ensemble_method'] == 'soft_voting':
            final_proba = all_predictions.mean(axis=1)
        elif self.config['ensemble_method'] == 'weighted_voting':
            final_proba = np.average(all_predictions, axis=1, weights=self.weights)
        elif self.config['ensemble_method'] == 'median':
            final_proba = np.median(all_predictions, axis=1)
        else:
            raise ValueError(f"Unknown ensemble method: {self.config['ensemble_method']}")
        
        # Return as (n_samples, 2) for consistency
        result = np.column_stack([1 - final_proba, final_proba])
        
        return result
    
    def _create_feature_groups(self, X, y, feature_names):
        """Create feature groups based on strategy"""
        n_features = len(feature_names)
        
        if n_features <= self.max_features:
            print(f"  Features within limit ({n_features} <= {self.max_features}), using all features")
            return [feature_names]
        
        strategy = self.config['feature_strategy']
        
        print(f"  Features exceed limit ({n_features} > {self.max_features})")
        print(f"  Using feature strategy: {strategy}")
        
        if strategy == 'random_groups':
            n_groups = self.config.get('n_feature_groups', 3)
            return FeatureSelector.random_groups(
                feature_names, n_groups, 
                overlap=self.config.get('feature_overlap', 0.1),
                random_state=self.config['random_state'],
                max_features_per_group=self.max_features
            )
        elif strategy == 'all':
            raise ValueError(f"Cannot use 'all' strategy with {n_features} features (limit: {self.max_features})")
        else:
            raise ValueError(f"Unknown feature strategy: {strategy}")
    
    def _create_sample_bags(self, X, y):
        """Create sample bags based on strategy"""
        strategy = self.config['bagging_strategy']
        n_bags = self.config['n_bags']
        bag_size = min(self.config['bag_sample_size'], self.max_samples, len(X))
        
        print(f"  Creating {n_bags} sample bags of size {bag_size:,}...")
        
        if strategy == 'stratified_random':
            bags = SamplingStrategy.stratified_random(
                X, y, n_bags, bag_size,
                overlap=self.config['bag_overlap'],
                random_state=self.config['random_state']
            )
        elif strategy == 'bootstrap':
            bags = SamplingStrategy.bootstrap(
                X, y, n_bags, bag_size,
                random_state=self.config['random_state']
            )
        elif strategy == 'diversity':
            bags = SamplingStrategy.diversity(
                X, y, n_bags, bag_size,
                random_state=self.config['random_state']
            )
        else:
            raise ValueError(f"Unknown bagging strategy: {strategy}")
        
        # Print bag statistics
        for i, (train_idx, oob_idx) in enumerate(bags):
            print(f"    Bag {i+1}: {len(train_idx):,} train, {len(oob_idx):,} OOB")
        
        return bags
    
    def _compute_weights(self):
        """Compute model weights based on OOB scores"""
        if not self.config['use_oob_weighting']:
            self.weights = np.ones(len(self.models)) / len(self.models)
            return
        
        # Use OOB scores as weights
        oob_scores = np.array(self.oob_scores)
        
        # Handle None/NaN scores
        oob_scores = np.nan_to_num(oob_scores, nan=0.5)
        
        # Normalize to weights (softmax-style)
        # Better models get exponentially more weight
        weights = np.exp(oob_scores * 5)  # Temperature = 5
        weights = weights / weights.sum()
        
        self.weights = weights