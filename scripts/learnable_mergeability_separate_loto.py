#!/usr/bin/env python3
"""
Leave-One-Task-Out Cross-Validation for Mergeability Prediction.

This script trains separate MLPs using LOTO CV: for each of the 20 tasks,
train on 19 tasks and validate on 1 held-out task. Aggregate results across
all folds for robust performance estimation.
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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


def train_single_fold(train_indices, val_indices, metrics_normalized,
                      performance_normalized, cfg_learn, device, verbose=False):
    """
    Train a single fold (one model with given train/val split).

    Returns:
        dict with train/val predictions, targets, and best validation loss
    """
    input_dim = metrics_normalized.shape[1]

    # Create datasets
    train_dataset = SingleMethodDataset(
        metrics_normalized[train_indices],
        performance_normalized[train_indices]
    )
    val_dataset = SingleMethodDataset(
        metrics_normalized[val_indices],
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
            # Save best predictions
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
    output_dir = Path(cfg_learn.output_path) / 'loto_cv'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build metrics path
    metrics_path = Path(cfg.mergeability.output_path) / f"pairwise_metrics_{cfg.mergeability.benchmark_name}.json"

    print("="*70)
    print("Leave-One-Task-Out Cross-Validation for Mergeability Prediction")
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

    # Normalize metrics (using all data for consistent normalization)
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
        print(f"LOTO Cross-Validation for: {method}")
        print("="*70)

        # Extract performance for this method
        perf_normalized = performance_normalized[:, method_idx]

        # Storage for all folds
        all_folds_train_preds = []
        all_folds_train_targets = []
        all_folds_val_preds = []
        all_folds_val_targets = []
        fold_results = []

        # Leave-one-task-out loop
        for fold_idx, held_out_task in enumerate(all_tasks):
            print(f"\nFold {fold_idx+1}/{n_tasks}: Held-out task = {held_out_task}")

            # Determine train and validation tasks
            train_tasks = [t for t in all_tasks if t != held_out_task]
            val_tasks = [held_out_task]

            # Split pairs based on task membership
            train_indices = []
            val_indices = []

            for i, pair_name in enumerate(pair_names):
                task1, task2 = pair_name.split('__')

                # Training: both tasks in train_tasks
                if task1 in train_tasks and task2 in train_tasks:
                    train_indices.append(i)
                # Validation: at least one task is held-out
                elif task1 == held_out_task or task2 == held_out_task:
                    val_indices.append(i)

            train_indices = np.array(train_indices)
            val_indices = np.array(val_indices)

            print(f"  Train pairs: {len(train_indices)}")
            print(f"  Val pairs: {len(val_indices)}")

            if len(val_indices) == 0:
                print(f"  WARNING: No validation pairs for {held_out_task}, skipping")
                continue

            # Train this fold
            fold_result = train_single_fold(
                train_indices, val_indices,
                metrics_normalized, perf_normalized,
                cfg_learn, cfg_learn.device,
                verbose=(fold_idx == 0)  # Print details for first fold only
            )

            # Denormalize predictions and targets
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

            # Compute metrics for this fold
            train_metrics = compute_metrics_single(train_preds, train_targets)
            val_metrics = compute_metrics_single(val_preds, val_targets)

            print(f"  Train r: {train_metrics['pearson_r']:.4f}, Val r: {val_metrics['pearson_r']:.4f}")

            # Store fold results
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
                'val_metrics': val_metrics,
                'best_val_loss': fold_result['best_val_loss']
            })

        # Aggregate results across all folds
        print(f"\n{'='*70}")
        print(f"Aggregate Results for {method}")
        print(f"{'='*70}")

        # Concatenate all predictions
        all_train_preds = np.concatenate(all_folds_train_preds, axis=0)
        all_train_targets = np.concatenate(all_folds_train_targets, axis=0)
        all_val_preds = np.concatenate(all_folds_val_preds, axis=0)
        all_val_targets = np.concatenate(all_folds_val_targets, axis=0)

        # Compute aggregate metrics
        aggregate_train_metrics = compute_metrics_single(all_train_preds, all_train_targets)
        aggregate_val_metrics = compute_metrics_single(all_val_preds, all_val_targets)

        print(f"\nAggregate Training Metrics (across all folds):")
        print(f"  Pearson r: {aggregate_train_metrics['pearson_r']:.4f}")
        print(f"  R2: {aggregate_train_metrics['r2']:.4f}")
        print(f"  MSE: {aggregate_train_metrics['mse']:.6f}")
        print(f"  MAE: {aggregate_train_metrics['mae']:.6f}")

        print(f"\nAggregate Validation Metrics (across all folds):")
        print(f"  Pearson r: {aggregate_val_metrics['pearson_r']:.4f}")
        print(f"  R2: {aggregate_val_metrics['r2']:.4f}")
        print(f"  MSE: {aggregate_val_metrics['mse']:.6f}")
        print(f"  MAE: {aggregate_val_metrics['mae']:.6f}")

        # Compute mean and std across folds
        fold_train_r = [f['train_metrics']['pearson_r'] for f in fold_results]
        fold_val_r = [f['val_metrics']['pearson_r'] for f in fold_results]

        print(f"\nPer-fold Statistics:")
        print(f"  Train r: {np.mean(fold_train_r):.4f} ± {np.std(fold_train_r):.4f}")
        print(f"  Val r:   {np.mean(fold_val_r):.4f} ± {np.std(fold_val_r):.4f}")
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
            'fold_results': fold_results,
            'model_config': {
                'input_dim': len(metric_names),
                'hidden_dim': cfg_learn.hidden_dim,
                'dropout': cfg_learn.dropout,
                'weight_decay': cfg_learn.get('weight_decay', 0.0),
                'n_parameters': 249  # 29*8 + 8 + 8*1 + 1
            }
        }

    # Save results
    results_file = output_dir / 'loto_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to: {results_file}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: LOTO Cross-Validation Results")
    print("="*70)
    print(f"{'Method':<15} {'Train r':<12} {'Val r':<12} {'Val r std':<12}")
    print("-"*70)
    for method in merge_methods:
        train_r = all_results[method]['aggregate_metrics']['train']['pearson_r']
        val_r = all_results[method]['aggregate_metrics']['validation']['pearson_r']
        val_r_std = all_results[method]['per_fold_stats']['val_r_std']
        print(f"{method:<15} {train_r:<12.4f} {val_r:<12.4f} {val_r_std:<12.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
