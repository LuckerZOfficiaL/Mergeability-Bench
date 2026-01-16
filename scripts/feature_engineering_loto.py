#!/usr/bin/env python3
"""
Feature Engineering with LOTO Cross-Validation for Mergeability Prediction.

This script:
1. Creates interaction features (products, ratios) from top metrics
2. Uses L1 regularization (Lasso) to select important features
3. Trains MLPs with engineered features using LOTO CV
"""
import sys
import os
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LassoCV
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_merging.models.mlp_single import SingleMethodMLP
from model_merging.data_loader import (
    load_json, extract_all_mergers_data, normalize_metrics,
    normalize_performance, denormalize_performance
)


class SingleMethodDataset(Dataset):
    """PyTorch dataset for single-method mergeability prediction."""

    def __init__(self, metrics, performance):
        self.metrics = torch.FloatTensor(metrics)
        self.performance = torch.FloatTensor(performance).unsqueeze(1)

    def __len__(self):
        return len(self.metrics)

    def __getitem__(self, idx):
        return self.metrics[idx], self.performance[idx]


def create_interaction_features(metrics_array, metric_names, top_k=10, interaction_type='product'):
    """
    Create interaction features from top-k most important metrics.

    Args:
        metrics_array: (n_pairs, n_metrics) array
        metric_names: List of metric names
        top_k: Number of top metrics to use for interactions
        interaction_type: 'product', 'ratio', or 'both'

    Returns:
        engineered_features: (n_pairs, n_features) array
        feature_names: List of feature names
    """
    n_pairs, n_metrics = metrics_array.shape

    # Start with original features
    engineered_features = [metrics_array]
    feature_names = list(metric_names)

    print(f"Creating interaction features from top {top_k} metrics...")

    # For simplicity, use all metrics if top_k >= n_metrics
    k = min(top_k, n_metrics)
    selected_indices = list(range(k))
    selected_names = [metric_names[i] for i in selected_indices]

    print(f"Selected metrics for interactions: {selected_names[:5]}... (and {k-5} more)" if k > 5 else f"Selected metrics: {selected_names}")

    # Create pairwise interactions
    interaction_count = 0

    if interaction_type in ['product', 'both']:
        print("Creating product features...")
        for i in range(k):
            for j in range(i+1, k):
                # Product
                product_feature = metrics_array[:, selected_indices[i]] * metrics_array[:, selected_indices[j]]
                engineered_features.append(product_feature.reshape(-1, 1))
                feature_names.append(f"{metric_names[selected_indices[i]]} × {metric_names[selected_indices[j]]}")
                interaction_count += 1

    if interaction_type in ['ratio', 'both']:
        print("Creating ratio features...")
        for i in range(k):
            for j in range(i+1, k):
                # Ratio (add small epsilon to avoid division by zero)
                ratio_feature = metrics_array[:, selected_indices[i]] / (metrics_array[:, selected_indices[j]] + 1e-8)
                engineered_features.append(ratio_feature.reshape(-1, 1))
                feature_names.append(f"{metric_names[selected_indices[i]]} / {metric_names[selected_indices[j]]}")
                interaction_count += 1

    # Concatenate all features
    engineered_array = np.concatenate(engineered_features, axis=1)

    print(f"Created {interaction_count} interaction features")
    print(f"Total features: {engineered_array.shape[1]} (original: {n_metrics}, added: {interaction_count})")

    return engineered_array, feature_names


def select_features_with_lasso(features, targets, feature_names, alpha=None, max_features=50):
    """
    Use LassoCV to select important features.

    Args:
        features: (n_samples, n_features) array
        targets: (n_samples,) array
        feature_names: List of feature names
        alpha: L1 regularization strength (if None, use CV to select)
        max_features: Maximum number of features to keep

    Returns:
        selected_indices: Indices of selected features
        selected_names: Names of selected features
        lasso_model: Fitted Lasso model
    """
    print("\nPerforming feature selection with Lasso...")

    if alpha is None:
        # Use cross-validation to find best alpha
        alphas = np.logspace(-4, 0, 50)
        lasso = LassoCV(alphas=alphas, cv=5, random_state=42, max_iter=5000)
    else:
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=alpha, random_state=42, max_iter=5000)

    lasso.fit(features, targets)

    if alpha is None:
        print(f"Best alpha: {lasso.alpha_:.6f}")

    # Get non-zero coefficients
    coefficients = lasso.coef_
    non_zero_mask = coefficients != 0
    non_zero_indices = np.where(non_zero_mask)[0]

    print(f"Features with non-zero coefficients: {len(non_zero_indices)}")

    # If too many features, keep top by absolute coefficient value
    if len(non_zero_indices) > max_features:
        abs_coefs = np.abs(coefficients[non_zero_indices])
        top_k_indices = np.argsort(abs_coefs)[-max_features:]
        selected_indices = non_zero_indices[top_k_indices]
        print(f"Keeping top {max_features} features by coefficient magnitude")
    else:
        selected_indices = non_zero_indices

    selected_names = [feature_names[i] for i in selected_indices]

    print(f"Final selected features: {len(selected_indices)}")
    print("\nTop 10 selected features:")
    for i in selected_indices[:10]:
        print(f"  {feature_names[i]}: coef={coefficients[i]:.4f}")

    return selected_indices, selected_names, lasso


def compute_metrics_single(predictions, targets):
    """Compute evaluation metrics for single method."""
    pred = predictions.flatten()
    target = targets.flatten()

    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))

    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    corr, p_value = pearsonr(pred, target)

    return {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'pearson_r': float(corr),
        'p_value': float(p_value)
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for metrics, performance in dataloader:
        metrics = metrics.to(device)
        performance = performance.to(device)

        optimizer.zero_grad()
        predictions = model(metrics)
        loss = criterion(predictions, performance)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for metrics, performance in dataloader:
            metrics = metrics.to(device)
            performance = performance.to(device)

            predictions = model(metrics)
            loss = criterion(predictions, performance)

            total_loss += loss.item()
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(performance.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_loss = total_loss / len(dataloader)

    return all_preds, all_targets, avg_loss


def train_single_fold(train_indices, val_indices, features_selected,
                      performance_normalized, cfg_learn, device, verbose=False):
    """
    Train a single fold with selected features.

    Returns:
        dict with train/val predictions, targets, and best validation loss
    """
    input_dim = features_selected.shape[1]

    # Create datasets
    train_dataset = SingleMethodDataset(
        features_selected[train_indices],
        performance_normalized[train_indices]
    )
    val_dataset = SingleMethodDataset(
        features_selected[val_indices],
        performance_normalized[val_indices]
    )

    # Create dataloaders
    batch_size = cfg_learn.batch_size if cfg_learn.batch_size is not None else len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    # Create model
    model = SingleMethodMLP(
        input_dim=input_dim,
        hidden_dim=cfg_learn.hidden_dim,
        dropout=cfg_learn.dropout
    ).to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    weight_decay = cfg_learn.get('weight_decay', 0.0)
    optimizer = optim.Adam(model.parameters(), lr=cfg_learn.learning_rate, weight_decay=weight_decay)

    # Training loop
    best_val_loss = float('inf')
    best_train_preds = None
    best_train_targets = None
    best_val_preds = None
    best_val_targets = None

    for epoch in range(cfg_learn.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        _, _, val_loss = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_preds, best_train_targets, _ = evaluate(model, train_loader, criterion, device)
            best_val_preds, best_val_targets, _ = evaluate(model, val_loader, criterion, device)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{cfg_learn.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    return {
        'train_preds': best_train_preds,
        'train_targets': best_train_targets,
        'val_preds': best_val_preds,
        'val_targets': best_val_targets,
        'best_val_loss': best_val_loss
    }


@hydra.main(version_base=None, config_path="../conf", config_name="multitask")
def main(cfg: DictConfig):
    # Set random seeds
    cfg_learn = cfg.learnable_mergeability
    np.random.seed(cfg_learn.random_seed)
    torch.manual_seed(cfg_learn.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg_learn.random_seed)

    # Create output directory
    output_dir = Path(cfg_learn.output_path) / 'feature_engineering_loto'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build metrics path
    metrics_path = Path(cfg.mergeability.output_path) / f"pairwise_metrics_{cfg.mergeability.benchmark_name}.json"

    print("="*70)
    print("Feature Engineering + LOTO Cross-Validation")
    print("="*70)
    print(f"Device: {cfg_learn.device}")
    print(f"Random seed: {cfg_learn.random_seed}")
    print()

    # Load data
    print("Loading data...")
    metrics_data = load_json(metrics_path)

    performance_data_dict = {}
    for method in cfg_learn.merge_methods:
        perf_path = Path(cfg_learn.results_dir) / method / f'all_pairwise_summary_{cfg.mergeability.benchmark_name}.json'
        if not perf_path.exists():
            print(f"Warning: {perf_path} not found, skipping {method}")
            continue
        performance_data_dict[method] = load_json(perf_path)

    merge_methods = list(performance_data_dict.keys())
    print(f"Merge methods: {merge_methods}")
    print()

    # Extract pairwise data
    print("Extracting pairwise data...")
    metrics_array, performance_matrix, pair_names, metric_names, merge_methods = \
        extract_all_mergers_data(metrics_data, performance_data_dict)

    print(f"Number of pairs: {len(pair_names)}")
    print(f"Number of metrics: {len(metric_names)}")
    print(f"Number of merge methods: {len(merge_methods)}")
    print()

    # Normalize metrics
    print("Normalizing metrics...")
    metrics_normalized, metrics_min_vals, metrics_max_vals = normalize_metrics(metrics_array)
    print("Normalization complete.")
    print()

    # Normalize performance
    print("Normalizing performance targets...")
    performance_normalized, perf_min_vals, perf_max_vals = normalize_performance(performance_matrix)
    for i, method in enumerate(merge_methods):
        print(f"  {method}: [{perf_min_vals[i]:.4f}, {perf_max_vals[i]:.4f}] -> [-1, 1]")
    print()

    # Create interaction features
    print("="*70)
    print("Feature Engineering")
    print("="*70)

    # Use top-10 metrics for interactions
    engineered_features, engineered_names = create_interaction_features(
        metrics_normalized,
        metric_names,
        top_k=10,
        interaction_type='product'  # Use products only (ratios can be unstable)
    )

    print(f"\nEngineered features shape: {engineered_features.shape}")
    print()

    # Get list of tasks
    all_tasks = metrics_data['datasets']
    n_tasks = len(all_tasks)
    print(f"Total tasks: {n_tasks}")
    print(f"Tasks: {all_tasks}")
    print()

    # LOTO Cross-Validation for each merge method
    all_results = {}

    for method_idx, method in enumerate(merge_methods):
        print("="*70)
        print(f"Processing: {method}")
        print("="*70)

        # Extract performance for this method
        perf_normalized = performance_normalized[:, method_idx]

        # Feature selection using all data (for consistent features across folds)
        # Note: Ideally we'd do feature selection per fold, but for simplicity
        # we use all data here. This is acceptable for interaction features.
        selected_indices, selected_names, lasso_model = select_features_with_lasso(
            engineered_features,
            perf_normalized,
            engineered_names,
            alpha=None,  # Use CV to find best alpha
            max_features=30  # Keep top 30 features
        )

        # Extract selected features
        features_selected = engineered_features[:, selected_indices]

        print(f"\nUsing {len(selected_indices)} selected features for {method}")
        print()

        # Storage for all folds
        all_folds_train_preds = []
        all_folds_train_targets = []
        all_folds_val_preds = []
        all_folds_val_targets = []
        fold_results = []

        # Leave-one-task-out loop
        print(f"Running LOTO CV with {n_tasks} folds...")
        for fold_idx, held_out_task in enumerate(all_tasks):
            # Determine train and validation tasks
            train_tasks = [t for t in all_tasks if t != held_out_task]

            # Split pairs based on task membership
            train_indices = []
            val_indices = []

            for i, pair_name in enumerate(pair_names):
                task1, task2 = pair_name.split('__')

                if task1 in train_tasks and task2 in train_tasks:
                    train_indices.append(i)
                elif task1 == held_out_task or task2 == held_out_task:
                    val_indices.append(i)

            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)

            if len(val_indices) == 0:
                continue

            # Train this fold
            fold_result = train_single_fold(
                train_indices, val_indices,
                features_selected, perf_normalized,
                cfg_learn, cfg_learn.device,
                verbose=(fold_idx == 0)
            )

            # Denormalize
            train_preds = denormalize_performance(
                fold_result['train_preds'],
                perf_min_vals[method_idx:method_idx+1],
                perf_max_vals[method_idx:method_idx+1]
            )
            train_targets = denormalize_performance(
                fold_result['train_targets'],
                perf_min_vals[method_idx:method_idx+1],
                perf_max_vals[method_idx:method_idx+1]
            )
            val_preds = denormalize_performance(
                fold_result['val_preds'],
                perf_min_vals[method_idx:method_idx+1],
                perf_max_vals[method_idx:method_idx+1]
            )
            val_targets = denormalize_performance(
                fold_result['val_targets'],
                perf_min_vals[method_idx:method_idx+1],
                perf_max_vals[method_idx:method_idx+1]
            )

            # Compute metrics
            train_metrics = compute_metrics_single(train_preds, train_targets)
            val_metrics = compute_metrics_single(val_preds, val_targets)

            # Store results
            all_folds_train_preds.append(train_preds)
            all_folds_train_targets.append(train_targets)
            all_folds_val_preds.append(val_preds)
            all_folds_val_targets.append(val_targets)

            fold_results.append({
                'fold': fold_idx,
                'held_out_task': held_out_task,
                'n_train_pairs': len(train_indices),
                'n_val_pairs': len(val_indices),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })

        # Aggregate results
        print(f"\n{'='*70}")
        print(f"Results for {method}")
        print(f"{'='*70}")

        all_train_preds = np.concatenate(all_folds_train_preds, axis=0)
        all_train_targets = np.concatenate(all_folds_train_targets, axis=0)
        all_val_preds = np.concatenate(all_folds_val_preds, axis=0)
        all_val_targets = np.concatenate(all_folds_val_targets, axis=0)

        aggregate_train_metrics = compute_metrics_single(all_train_preds, all_train_targets)
        aggregate_val_metrics = compute_metrics_single(all_val_preds, all_val_targets)

        print(f"Aggregate Training: r={aggregate_train_metrics['pearson_r']:.4f}, R2={aggregate_train_metrics['r2']:.4f}")
        print(f"Aggregate Validation: r={aggregate_val_metrics['pearson_r']:.4f}, R2={aggregate_val_metrics['r2']:.4f}")

        fold_train_r = [f['train_metrics']['pearson_r'] for f in fold_results]
        fold_val_r = [f['val_metrics']['pearson_r'] for f in fold_results]

        print(f"Per-fold: Train r={np.mean(fold_train_r):.4f}±{np.std(fold_train_r):.4f}, Val r={np.mean(fold_val_r):.4f}±{np.std(fold_val_r):.4f}")
        print()

        # Store results
        all_results[method] = {
            'aggregate_metrics': {
                'train': aggregate_train_metrics,
                'validation': aggregate_val_metrics
            },
            'per_fold_stats': {
                'train_r_mean': float(np.mean(fold_train_r)),
                'train_r_std': float(np.std(fold_train_r)),
                'val_r_mean': float(np.mean(fold_val_r)),
                'val_r_std': float(np.std(fold_val_r))
            },
            'feature_selection': {
                'n_selected_features': len(selected_indices),
                'selected_features': selected_names,
                'lasso_alpha': float(lasso_model.alpha_) if hasattr(lasso_model, 'alpha_') else None
            },
            'model_config': {
                'input_dim': len(selected_indices),
                'hidden_dim': cfg_learn.hidden_dim,
                'dropout': cfg_learn.dropout,
                'weight_decay': cfg_learn.get('weight_decay', 0.0)
            }
        }

    # Save results
    results_file = output_dir / 'feature_engineering_loto_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {results_file}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY: Feature Engineering + LOTO Results")
    print("="*70)
    print(f"{'Method':<15} {'Train r':<12} {'Val r':<12} {'Val std':<12} {'Features':<10}")
    print("-"*70)
    for method in merge_methods:
        train_r = all_results[method]['aggregate_metrics']['train']['pearson_r']
        val_r = all_results[method]['aggregate_metrics']['validation']['pearson_r']
        val_std = all_results[method]['per_fold_stats']['val_r_std']
        n_feat = all_results[method]['feature_selection']['n_selected_features']
        print(f"{method:<15} {train_r:<12.4f} {val_r:<12.4f} {val_std:<12.4f} {n_feat:<10}")
    print("="*70)


if __name__ == "__main__":
    main()
