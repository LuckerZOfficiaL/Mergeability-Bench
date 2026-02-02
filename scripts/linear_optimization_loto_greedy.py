#!/usr/bin/env python3
"""
Greedy Forward Selection for LOTO Cross-Validation.

This script performs greedy metric selection within each fold:
- Start with empty set, iteratively add the metric that improves training correlation most
- Stop when improvement < threshold (default 0.01)
- Validation data is only used for final evaluation (no leakage)
- Each fold may select different metrics

Usage:
    python scripts/linear_optimization_loto_greedy.py
    python scripts/linear_optimization_loto_greedy.py --threshold 0.02
"""
import sys
from pathlib import Path
import json
import numpy as np
import torch
from scipy.stats import pearsonr
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_merging.data_loader import load_json, extract_all_mergers_data


def normalize_metrics(metrics_array):
    """Normalize metrics to [-1, 1] range using min-max normalization."""
    min_vals = metrics_array.min(axis=0)
    max_vals = metrics_array.max(axis=0)

    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    normalized = (metrics_array - min_vals) / ranges
    normalized = normalized * 2 - 1

    return normalized, min_vals, max_vals


def normalize_metrics_with_stats(metrics_array, min_vals, max_vals):
    """Normalize metrics using pre-computed min/max values (for validation data)."""
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    normalized = (metrics_array - min_vals) / ranges
    normalized = normalized * 2 - 1

    return normalized


def optimize_coefficients(metrics_train, performance_train,
                          n_iterations=1000, lr=0.01,
                          patience=50, convergence_threshold=1e-4):
    """
    Optimize linear coefficients on training data only.

    Returns:
        coefficients: Optimized coefficients
        train_r: Training Pearson r
    """
    n_metrics = metrics_train.shape[1]

    if n_metrics == 0:
        return np.array([]), 0.0

    # Initialize coefficients
    coefficients = torch.randn(n_metrics, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([coefficients], lr=lr)

    # Convert to tensors
    X_train = torch.FloatTensor(metrics_train)
    y_train = torch.FloatTensor(performance_train)

    best_train_loss = float('inf')
    patience_counter = 0

    for iteration in range(n_iterations):
        optimizer.zero_grad()

        # Forward pass
        predictions = X_train @ coefficients

        # Loss: negative Pearson correlation
        pred_mean = predictions.mean()
        target_mean = y_train.mean()

        pred_centered = predictions - pred_mean
        target_centered = y_train - target_mean

        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())

        correlation = numerator / (denominator + 1e-8)
        loss = -correlation

        # Backward pass
        loss.backward()

        # Constraint: coefficients sum to 1
        with torch.no_grad():
            coefficients.data = coefficients.data / (coefficients.data.sum() + 1e-8)

        optimizer.step()

        # Early stopping
        if loss.item() < best_train_loss - convergence_threshold:
            best_train_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Final evaluation on training data
    with torch.no_grad():
        train_pred = (X_train @ coefficients).numpy()

    train_r, _ = pearsonr(train_pred, performance_train)

    return coefficients.detach().numpy(), train_r


def greedy_forward_selection(metrics_train, performance_train, metric_names,
                              threshold=0.01, verbose=False):
    """
    Greedy forward selection of metrics based on training correlation.

    Args:
        metrics_train: Training metrics array (n_samples, n_metrics)
        performance_train: Training target values
        metric_names: List of metric names
        threshold: Minimum improvement to continue adding metrics
        verbose: Print progress

    Returns:
        selected_indices: List of selected metric indices in order
        selected_names: List of selected metric names in order
        final_coefficients: Coefficients for selected metrics
        final_train_r: Final training correlation
        selection_history: List of (metric_name, train_r, improvement) for each round
    """
    n_metrics = len(metric_names)
    available_indices = list(range(n_metrics))
    selected_indices = []
    selection_history = []

    current_train_r = 0.0

    while available_indices:
        best_metric_idx = None
        best_train_r = current_train_r
        best_improvement = 0.0

        # Try adding each available metric
        for candidate_idx in available_indices:
            # Create candidate set
            candidate_indices = selected_indices + [candidate_idx]
            candidate_metrics = metrics_train[:, candidate_indices]

            # Optimize and evaluate
            _, train_r = optimize_coefficients(candidate_metrics, performance_train)

            improvement = train_r - current_train_r
            if train_r > best_train_r:
                best_train_r = train_r
                best_metric_idx = candidate_idx
                best_improvement = improvement

        # Check if improvement meets threshold
        if best_improvement < threshold:
            if verbose:
                print(f"    Stopping: best improvement {best_improvement:.4f} < threshold {threshold}")
            break

        # Add best metric
        selected_indices.append(best_metric_idx)
        available_indices.remove(best_metric_idx)
        current_train_r = best_train_r

        selection_history.append({
            'round': len(selected_indices),
            'metric': metric_names[best_metric_idx],
            'train_r': float(best_train_r),
            'improvement': float(best_improvement)
        })

        if verbose:
            print(f"    Round {len(selected_indices)}: Added '{metric_names[best_metric_idx]}' "
                  f"(train_r={best_train_r:.4f}, Δ={best_improvement:+.4f})")

    # Get final coefficients for selected metrics
    if selected_indices:
        selected_metrics = metrics_train[:, selected_indices]
        final_coefficients, final_train_r = optimize_coefficients(selected_metrics, performance_train)
    else:
        final_coefficients = np.array([])
        final_train_r = 0.0

    selected_names = [metric_names[i] for i in selected_indices]

    return selected_indices, selected_names, final_coefficients, final_train_r, selection_history


def run_loto_cv_greedy(metrics_array, performance_array, pair_names, all_tasks,
                       metric_names, threshold=0.01):
    """
    Run Leave-One-Task-Out cross-validation with greedy metric selection.

    Returns:
        results: Dictionary with fold results and aggregate metrics
    """
    n_tasks = len(all_tasks)
    n_metrics = len(metric_names)

    # Storage for results
    fold_results = []
    all_train_preds = []
    all_train_targets = []
    all_val_preds = []
    all_val_targets = []

    # Track coefficients across folds (with 0 for unselected)
    fold_full_coefficients = []

    # Track selection frequency
    selection_counts = {name: 0 for name in metric_names}

    print(f"Running Greedy LOTO CV with {n_tasks} folds (threshold={threshold})...")
    print()

    for fold_idx, held_out_task in enumerate(all_tasks):
        print(f"Fold {fold_idx+1}/{n_tasks}: Held-out task = {held_out_task}")

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
            print(f"  WARNING: No validation pairs for {held_out_task}, skipping")
            continue

        print(f"  Train pairs: {len(train_indices)}, Val pairs: {len(val_indices)}")

        # Extract raw data for this fold
        metrics_train_raw = metrics_array[train_indices]
        performance_train = performance_array[train_indices]
        metrics_val_raw = metrics_array[val_indices]
        performance_val = performance_array[val_indices]

        # Normalize using ONLY training data statistics (no leakage)
        metrics_train, min_vals, max_vals = normalize_metrics(metrics_train_raw)
        metrics_val = normalize_metrics_with_stats(metrics_val_raw, min_vals, max_vals)

        # Greedy forward selection on TRAINING data only
        selected_indices, selected_names, selected_coefficients, train_r, selection_history = \
            greedy_forward_selection(metrics_train, performance_train, metric_names,
                                     threshold=threshold, verbose=True)

        print(f"  Selected {len(selected_names)} metrics, train_r={train_r:.4f}")

        # Evaluate on validation data (only once, no leakage)
        if selected_indices:
            selected_metrics_train = metrics_train[:, selected_indices]
            selected_metrics_val = metrics_val[:, selected_indices]
            train_preds = selected_metrics_train @ selected_coefficients
            val_preds = selected_metrics_val @ selected_coefficients
            val_r, _ = pearsonr(val_preds, performance_val)
        else:
            train_preds = np.zeros(len(performance_train))
            val_preds = np.zeros(len(performance_val))
            val_r = 0.0

        print(f"  Validation r={val_r:.4f}")

        # Store predictions for aggregate evaluation
        all_train_preds.append(train_preds)
        all_train_targets.append(performance_train)
        all_val_preds.append(val_preds)
        all_val_targets.append(performance_val)

        # Create full coefficient vector (0 for unselected metrics)
        full_coefficients = np.zeros(n_metrics)
        for idx, coef in zip(selected_indices, selected_coefficients):
            full_coefficients[idx] = coef
        fold_full_coefficients.append(full_coefficients)

        # Update selection counts
        for name in selected_names:
            selection_counts[name] += 1

        # Store fold results
        fold_results.append({
            'fold': fold_idx,
            'held_out_task': held_out_task,
            'n_train_pairs': len(train_indices),
            'n_val_pairs': len(val_indices),
            'train_r': float(train_r),
            'val_r': float(val_r),
            'n_selected_metrics': len(selected_names),
            'selected_metrics': selected_names,
            'selected_coefficients': {name: float(coef) for name, coef in zip(selected_names, selected_coefficients)},
            'selection_history': selection_history
        })

        print()

    # Aggregate results
    print("="*70)
    print("Aggregate Results")
    print("="*70)

    # Concatenate all predictions
    all_train_preds = np.concatenate(all_train_preds)
    all_train_targets = np.concatenate(all_train_targets)
    all_val_preds = np.concatenate(all_val_preds)
    all_val_targets = np.concatenate(all_val_targets)

    # Compute aggregate correlations
    aggregate_train_r, aggregate_train_p = pearsonr(all_train_preds, all_train_targets)
    aggregate_val_r, aggregate_val_p = pearsonr(all_val_preds, all_val_targets)

    print(f"Aggregate Training: r={aggregate_train_r:.4f}, p={aggregate_train_p:.2e}")
    print(f"Aggregate Validation: r={aggregate_val_r:.4f}, p={aggregate_val_p:.2e}")

    # Per-fold statistics
    fold_train_r = [f['train_r'] for f in fold_results]
    fold_val_r = [f['val_r'] for f in fold_results]
    fold_n_selected = [f['n_selected_metrics'] for f in fold_results]

    print(f"Per-fold: Train r={np.mean(fold_train_r):.4f}±{np.std(fold_train_r):.4f}")
    print(f"Per-fold: Val r={np.mean(fold_val_r):.4f}±{np.std(fold_val_r):.4f}")
    print(f"Per-fold: N selected={np.mean(fold_n_selected):.1f}±{np.std(fold_n_selected):.1f}")

    # Average coefficients across folds
    avg_coefficients = np.mean(fold_full_coefficients, axis=0)
    std_coefficients = np.std(fold_full_coefficients, axis=0)

    # Selection frequency (proportion of folds where each metric was selected)
    selection_frequency = {name: count / n_tasks for name, count in selection_counts.items()}

    results = {
        'aggregate_metrics': {
            'train_r': float(aggregate_train_r),
            'train_p': float(aggregate_train_p),
            'val_r': float(aggregate_val_r),
            'val_p': float(aggregate_val_p)
        },
        'per_fold_stats': {
            'train_r_mean': float(np.mean(fold_train_r)),
            'train_r_std': float(np.std(fold_train_r)),
            'val_r_mean': float(np.mean(fold_val_r)),
            'val_r_std': float(np.std(fold_val_r)),
            'n_selected_mean': float(np.mean(fold_n_selected)),
            'n_selected_std': float(np.std(fold_n_selected))
        },
        'average_coefficients': {name: float(coef) for name, coef in zip(metric_names, avg_coefficients)},
        'coefficient_std': {name: float(std) for name, std in zip(metric_names, std_coefficients)},
        'selection_frequency': selection_frequency,
        'fold_results': fold_results,
        'greedy_params': {
            'threshold': threshold
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Greedy Forward Selection LOTO Cross-Validation')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Minimum training correlation improvement to continue adding metrics')
    args = parser.parse_args()

    # Configuration
    metrics_path = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results/mergeability/ViT-B-16/pairwise_metrics_N20.json')
    results_base_path = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results/ViT-B-16')
    output_dir = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results/metric_linear_optimization/loto_cv_greedy_selection')
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_methods = ['weight_avg', 'arithmetic', 'tsv', 'isotropic']

    print("="*70)
    print(f"Greedy Forward Selection with LOTO CV (threshold={args.threshold})")
    print("="*70)
    print()

    # Load metrics data
    print("Loading data...")
    metrics_data = load_json(metrics_path)

    # Load performance data for all methods
    performance_data_dict = {}
    for method in merge_methods:
        perf_path = results_base_path / method / 'all_pairwise_summary_N20.json'
        if not perf_path.exists():
            print(f"Warning: {perf_path} not found, skipping {method}")
            continue
        performance_data_dict[method] = load_json(perf_path)

    print(f"Loaded data for methods: {list(performance_data_dict.keys())}")
    print()

    # Extract pairwise data
    print("Extracting pairwise data...")
    metrics_array, performance_matrix, pair_names, metric_names, merge_methods = \
        extract_all_mergers_data(metrics_data, performance_data_dict)

    print(f"Number of pairs: {len(pair_names)}")
    print(f"Number of metrics: {len(metric_names)}")
    print(f"Number of merge methods: {len(merge_methods)}")
    print()

    # Get list of tasks
    all_tasks = metrics_data['datasets']
    print(f"Total tasks: {len(all_tasks)}")
    print()

    # Run LOTO for each merge method
    all_results = {}

    for method_idx, method in enumerate(merge_methods):
        print("="*70)
        print(f"Greedy LOTO Cross-Validation for: {method}")
        print("="*70)
        print()

        # Extract performance for this method
        performance = performance_matrix[:, method_idx]

        # Run Greedy LOTO CV
        results = run_loto_cv_greedy(
            metrics_array,
            performance,
            pair_names,
            all_tasks,
            metric_names,
            threshold=args.threshold
        )

        all_results[method] = results

        # Save individual method results
        method_output_file = output_dir / f'{method}_loto_results.json'
        with open(method_output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to: {method_output_file}")
        print()

    # Save combined results
    combined_results = {
        'greedy_threshold': args.threshold,
        'methods': all_results
    }
    combined_output_file = output_dir / 'all_methods_loto_results.json'
    with open(combined_output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    # Print summary
    print("="*70)
    print("SUMMARY: Greedy Forward Selection LOTO Results")
    print("="*70)
    print()
    print(f"{'Method':<15} {'Train r':<12} {'Val r':<12} {'Val r std':<12} {'N selected':<12}")
    print("-"*70)
    for method in merge_methods:
        train_r = all_results[method]['per_fold_stats']['train_r_mean']
        val_r = all_results[method]['per_fold_stats']['val_r_mean']
        val_r_std = all_results[method]['per_fold_stats']['val_r_std']
        n_selected = all_results[method]['per_fold_stats']['n_selected_mean']
        print(f"{method:<15} {train_r:<12.4f} {val_r:<12.4f} {val_r_std:<12.4f} {n_selected:<12.1f}")
    print("="*70)

    # Print most frequently selected metrics
    print()
    print("="*70)
    print("Most Frequently Selected Metrics (across all folds)")
    print("="*70)

    for method in merge_methods:
        freq = all_results[method]['selection_frequency']
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{method.upper()}:")
        print("-" * 60)
        for metric, f in sorted_freq[:10]:
            if f > 0:
                print(f"  {metric:<45} {f:>6.0%}")

    print()
    print("="*70)
    print("Greedy Forward Selection Complete!")
    print("="*70)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
