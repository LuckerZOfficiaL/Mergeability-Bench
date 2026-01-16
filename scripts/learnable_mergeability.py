#!/usr/bin/env python3
"""
Train MLP to predict mergeability scores for multiple merge methods.

This script trains a 2-layer MLP that takes 23-dimensional mergeability metrics
as input and predicts merge performance for 4 different merge methods simultaneously.
"""
import sys
import os
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_merging.models.mlp import MergeabilityMLP
from model_merging.data_loader import (
    load_json, extract_all_mergers_data, normalize_metrics,
    normalize_performance, denormalize_performance,
    task_level_split, MergeabilityDataset
)


def compute_metrics(predictions, targets, merge_methods):
    """
    Compute evaluation metrics for each merge method.

    Args:
        predictions: (n_samples, n_methods) array
        targets: (n_samples, n_methods) array
        merge_methods: List of merge method names

    Returns:
        Dictionary of metrics per merge method
    """
    metrics = {}

    for i, method in enumerate(merge_methods):
        pred = predictions[:, i]
        target = targets[:, i]

        # MSE
        mse = np.mean((pred - target) ** 2)

        # MAE
        mae = np.mean(np.abs(pred - target))

        # R-squared
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Pearson correlation
        corr, p_value = pearsonr(pred, target)

        metrics[method] = {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'pearson_r': float(corr),
            'p_value': float(p_value)
        }

    return metrics


def plot_predictions(predictions, targets, merge_methods, output_dir, split_name):
    """
    Create scatter plots of predictions vs targets for each merge method.

    Args:
        predictions: (n_samples, n_methods) array
        targets: (n_samples, n_methods) array
        merge_methods: List of merge method names
        output_dir: Directory to save plots
        split_name: Name of the split ('train' or 'validation')
    """
    n_methods = len(merge_methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))

    if n_methods == 1:
        axes = [axes]

    for i, (method, ax) in enumerate(zip(merge_methods, axes)):
        pred = predictions[:, i]
        target = targets[:, i]

        # Scatter plot
        ax.scatter(target, pred, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)

        # Best fit line
        z = np.polyfit(target, pred, 1)
        p = np.poly1d(z)
        target_sorted = np.sort(target)
        ax.plot(target_sorted, p(target_sorted), 'r-', linewidth=2, alpha=0.7)

        # Diagonal line (perfect prediction)
        min_val = min(target.min(), pred.min())
        max_val = max(target.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)

        # Compute metrics
        corr, p_value = pearsonr(pred, target)
        mse = np.mean((pred - target) ** 2)

        # Labels and title
        ax.set_xlabel('Actual Performance', fontsize=11)
        ax.set_ylabel('Predicted Performance', fontsize=11)
        ax.set_title(f'{method}\nr={corr:.3f}, MSE={mse:.4f}', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f'{split_name}_predictions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved {split_name} predictions plot to: {output_path}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for metrics, performance in dataloader:
        metrics = metrics.to(device)
        performance = performance.to(device)

        # Forward pass
        predictions = model(metrics)
        loss = criterion(predictions, performance)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for metrics, performance in dataloader:
            metrics = metrics.to(device)
            performance = performance.to(device)

            predictions = model(metrics)
            loss = criterion(predictions, performance)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(performance.cpu().numpy())

            total_loss += loss.item()
            n_batches += 1

    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    avg_loss = total_loss / n_batches

    return predictions, targets, avg_loss


@hydra.main(version_base=None, config_path="../conf", config_name="multitask")
def main(cfg: DictConfig):
    # Get configuration
    cfg_learn = cfg.learnable_mergeability

    # Set random seeds
    torch.manual_seed(cfg_learn.random_seed)
    np.random.seed(cfg_learn.random_seed)

    # Create output directories
    output_dir = Path(cfg_learn.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = output_dir / 'figs'
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Build metrics path
    metrics_path = Path(cfg.mergeability.output_path) / f"pairwise_metrics_{cfg.mergeability.benchmark_name}.json"

    print("="*70)
    print("Learnable Mergeability Prediction with MLP")
    print("="*70)
    print(f"Device: {cfg_learn.device}")
    print(f"Random seed: {cfg_learn.random_seed}")
    print()

    # Load metrics data
    print("Loading data...")
    metrics_data = load_json(metrics_path)

    # Load performance data for all merge methods
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

    # Extract pairwise data for all merge methods
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

    # Normalize performance targets
    print("Normalizing performance targets...")
    performance_normalized, perf_min_vals, perf_max_vals = normalize_performance(performance_matrix)
    print(f"Performance normalization:")
    for i, method in enumerate(merge_methods):
        print(f"  {method}: [{perf_min_vals[i]:.4f}, {perf_max_vals[i]:.4f}] -> [-1, 1]")
    print()

    # Task-level train/validation split
    print(f"Performing task-level split (validation={cfg_learn.validation_split:.1%})...")
    datasets = metrics_data['datasets']
    train_indices, val_indices, train_tasks, val_tasks = task_level_split(
        pair_names, datasets, cfg_learn.validation_split, cfg_learn.random_seed
    )

    print(f"  Training tasks ({len(train_tasks)}): {sorted(train_tasks)}")
    print(f"  Validation tasks ({len(val_tasks)}): {sorted(val_tasks)}")
    print(f"  Training pairs: {len(train_indices)}")
    print(f"  Validation pairs: {len(val_indices)}")
    print()

    # Create datasets with NORMALIZED performance
    train_dataset = MergeabilityDataset(
        metrics_normalized[train_indices],
        performance_normalized[train_indices]
    )
    val_dataset = MergeabilityDataset(
        metrics_normalized[val_indices],
        performance_normalized[val_indices]
    )

    # Create dataloaders
    batch_size = cfg_learn.batch_size if cfg_learn.batch_size is not None else len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

    print(f"Batch size: {batch_size}")
    print()

    # Create model
    input_dim = len(metric_names)
    output_dim = len(merge_methods)

    model = MergeabilityMLP(
        input_dim=input_dim,
        hidden_dim=cfg_learn.hidden_dim,
        output_dim=output_dim,
        dropout=cfg_learn.dropout
    )
    model = model.to(cfg_learn.device)

    print("Model Architecture:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {cfg_learn.hidden_dim}")
    print(f"  Dropout: {cfg_learn.dropout}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Total parameters: {model.count_parameters()}")
    print()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg_learn.learning_rate)

    print(f"Training for {cfg_learn.epochs} epochs...")
    print(f"Learning rate: {cfg_learn.learning_rate}")
    print()

    # Training loop
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': []
    }

    for epoch in range(cfg_learn.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, cfg_learn.device)

        # Evaluate
        train_preds, train_targets, train_eval_loss = evaluate(model, train_loader, criterion, cfg_learn.device)
        val_preds, val_targets, val_loss = evaluate(model, val_loader, criterion, cfg_learn.device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch+1}/{cfg_learn.epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  * New best validation loss!")

        print()

    # Final evaluation
    print("="*70)
    print("Final Evaluation")
    print("="*70)

    train_preds_norm, train_targets_norm, _ = evaluate(model, train_loader, criterion, cfg_learn.device)
    val_preds_norm, val_targets_norm, _ = evaluate(model, val_loader, criterion, cfg_learn.device)

    # Denormalize predictions and targets for metric computation
    train_preds = denormalize_performance(train_preds_norm, perf_min_vals, perf_max_vals)
    train_targets = denormalize_performance(train_targets_norm, perf_min_vals, perf_max_vals)
    val_preds = denormalize_performance(val_preds_norm, perf_min_vals, perf_max_vals)
    val_targets = denormalize_performance(val_targets_norm, perf_min_vals, perf_max_vals)

    # Compute metrics on denormalized values
    train_metrics = compute_metrics(train_preds, train_targets, merge_methods)
    val_metrics = compute_metrics(val_preds, val_targets, merge_methods)

    # Print results
    print("\nTraining Set Performance:")
    print("-"*70)
    print(f"{'Method':<15} {'MSE':<12} {'MAE':<12} {'R2':<12} {'Pearson r':<12}")
    print("-"*70)
    for method in merge_methods:
        m = train_metrics[method]
        print(f"{method:<15} {m['mse']:<12.6f} {m['mae']:<12.6f} {m['r2']:<12.4f} {m['pearson_r']:<12.4f}")
    print()

    print("Validation Set Performance:")
    print("-"*70)
    print(f"{'Method':<15} {'MSE':<12} {'MAE':<12} {'R2':<12} {'Pearson r':<12}")
    print("-"*70)
    for method in merge_methods:
        m = val_metrics[method]
        print(f"{method:<15} {m['mse']:<12.6f} {m['mae']:<12.6f} {m['r2']:<12.4f} {m['pearson_r']:<12.4f}")
    print()

    # Generate plots (using denormalized values)
    print("Generating prediction plots...")
    plot_predictions(train_preds, train_targets, merge_methods, figs_dir, 'train')
    plot_predictions(val_preds, val_targets, merge_methods, figs_dir, 'validation')
    print()

    # Save results
    results = {
        'model': {
            'input_dim': input_dim,
            'hidden_dim': cfg_learn.hidden_dim,
            'dropout': cfg_learn.dropout,
            'output_dim': output_dim,
            'n_parameters': model.count_parameters()
        },
        'data': {
            'n_pairs_total': len(pair_names),
            'n_pairs_train': len(train_indices),
            'n_pairs_val': len(val_indices),
            'n_metrics': len(metric_names),
            'n_merge_methods': len(merge_methods),
            'merge_methods': merge_methods,
            'train_tasks': sorted(list(train_tasks)),
            'val_tasks': sorted(list(val_tasks))
        },
        'training': {
            'epochs': cfg_learn.epochs,
            'learning_rate': cfg_learn.learning_rate,
            'batch_size': batch_size,
            'random_seed': cfg_learn.random_seed,
            'final_train_loss': float(history['train_loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'best_val_loss': float(best_val_loss)
        },
        'metrics': {
            'train': train_metrics,
            'validation': val_metrics
        },
        'normalization': {
            'metrics': {
                'min_vals': {name: float(val) for name, val in zip(metric_names, metrics_min_vals)},
                'max_vals': {name: float(val) for name, val in zip(metric_names, metrics_max_vals)}
            },
            'performance': {
                'min_vals': {method: float(val) for method, val in zip(merge_methods, perf_min_vals)},
                'max_vals': {method: float(val) for method, val in zip(merge_methods, perf_max_vals)}
            }
        }
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")
    print()
    print("="*70)
    print("Training complete!")
    print("="*70)


if __name__ == "__main__":
    main()
