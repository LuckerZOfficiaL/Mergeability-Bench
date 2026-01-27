#!/usr/bin/env python3
"""
Leave-Two-Tasks-Out Cross-Validation for L1-Regularized Linear Optimization.

This script performs L2TO CV for linear mergeability prediction with L1 regularization:
- For each of 190 task pairs, hold out that pair for validation
- Training uses only pairs where NEITHER task is in the held-out pair
- L1 penalty encourages sparse coefficients
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


def linear_optimization_single_fold_l1(metrics_train, performance_train,
                                        metrics_val, performance_val,
                                        lambda_l1=1.0,
                                        n_iterations=1000, lr=0.01,
                                        patience=50, convergence_threshold=1e-4):
    """
    Optimize linear coefficients for a single fold with L1 regularization.

    Returns:
        coefficients: Optimized coefficients (sparse)
        train_r: Training Pearson r
        val_pred: Validation prediction (single value for L2TO)
        n_iters: Number of iterations run
        n_nonzero: Number of non-zero coefficients
    """
    n_metrics = metrics_train.shape[1]

    # Initialize coefficients
    coefficients = torch.randn(n_metrics, dtype=torch.float32) * 0.1
    coefficients.requires_grad_(True)
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

        # Compute Pearson correlation
        pred_mean = predictions.mean()
        target_mean = y_train.mean()

        pred_centered = predictions - pred_mean
        target_centered = y_train - target_mean

        numerator = (pred_centered * target_centered).sum()
        denominator = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum())

        correlation = numerator / (denominator + 1e-8)

        # Loss: negative correlation + L1 penalty
        l1_penalty = torch.abs(coefficients).sum()
        loss = -correlation + lambda_l1 * l1_penalty

        # Backward pass
        loss.backward()
        optimizer.step()

        # Early stopping
        if loss.item() < best_train_loss - convergence_threshold:
            best_train_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Apply soft thresholding
    final_coefficients = coefficients.detach().numpy()
    threshold = 1e-3
    final_coefficients[np.abs(final_coefficients) < threshold] = 0

    # Final evaluation
    with torch.no_grad():
        coef_tensor = torch.FloatTensor(final_coefficients)
        train_pred = (X_train @ coef_tensor).numpy()

        # Validation prediction (may be single sample)
        X_val = torch.FloatTensor(metrics_val)
        val_pred = (X_val @ coef_tensor).numpy()

    train_r, _ = pearsonr(train_pred, performance_train)
    n_nonzero = np.sum(final_coefficients != 0)

    return final_coefficients, train_r, val_pred, iteration + 1, n_nonzero


def run_l2to_cv(metrics_array, performance_array, pair_names,
                metric_names, lambda_l1=1.0, n_iterations=1000, lr=0.01,
                patience=50, convergence_threshold=1e-4):
    """
    Run Leave-Two-Tasks-Out cross-validation with L1 regularization.

    For each pair (A, B):
    - Validation: the pair (A, B) itself
    - Training: all pairs where neither task is A or B
    - Ignored: all other pairs containing A or B

    Returns:
        results: Dictionary with fold results and aggregate metrics
    """
    n_pairs = len(pair_names)
    n_metrics = len(metric_names)

    # Storage for results
    fold_results = []
    all_val_preds = []
    all_val_targets = []
    fold_coefficients = []
    fold_nonzero_counts = []

    print(f"Running L2TO CV with {n_pairs} folds (L1 lambda={lambda_l1})...")
    print()

    for fold_idx, held_out_pair in enumerate(pair_names):
        if fold_idx % 20 == 0:
            print(f"Processing fold {fold_idx+1}/{n_pairs}...")

        # Parse held-out tasks
        task_a, task_b = held_out_pair.split('__')

        # Split pairs
        train_indices = []
        val_index = None

        for i, pair_name in enumerate(pair_names):
            t1, t2 = pair_name.split('__')

            if pair_name == held_out_pair:
                # This is the validation pair
                val_index = i
            elif t1 != task_a and t1 != task_b and t2 != task_a and t2 != task_b:
                # Neither task is in held-out pair -> training
                train_indices.append(i)
            # Otherwise: pair contains A or B but is not (A,B) -> ignored

        train_indices = np.array(train_indices)

        if val_index is None:
            print(f"  WARNING: Could not find validation pair {held_out_pair}, skipping")
            continue

        # Extract raw data
        metrics_train_raw = metrics_array[train_indices]
        performance_train = performance_array[train_indices]
        metrics_val_raw = metrics_array[[val_index]]
        performance_val = performance_array[[val_index]]

        # Normalize using ONLY training data statistics
        metrics_train, min_vals, max_vals = normalize_metrics(metrics_train_raw)
        metrics_val = normalize_metrics_with_stats(metrics_val_raw, min_vals, max_vals)

        # Optimize coefficients
        coefficients, train_r, val_pred, n_iters, n_nonzero = linear_optimization_single_fold_l1(
            metrics_train, performance_train,
            metrics_val, performance_val,
            lambda_l1=lambda_l1,
            n_iterations=n_iterations,
            lr=lr,
            patience=patience,
            convergence_threshold=convergence_threshold
        )

        # Store results
        all_val_preds.append(val_pred[0])
        all_val_targets.append(performance_val[0])
        fold_coefficients.append(coefficients)
        fold_nonzero_counts.append(n_nonzero)

        # Store fold results
        fold_results.append({
            'fold': fold_idx,
            'held_out_pair': held_out_pair,
            'n_train_pairs': len(train_indices),
            'train_r': float(train_r),
            'val_pred': float(val_pred[0]),
            'val_target': float(performance_val[0]),
            'n_iterations': int(n_iters),
            'n_nonzero_coefficients': int(n_nonzero),
            'coefficients': {name: float(coef) for name, coef in zip(metric_names, coefficients)}
        })

    # Aggregate results
    print()
    print("="*70)
    print("Aggregate Results")
    print("="*70)

    # Compute aggregate correlation across all held-out pairs
    all_val_preds = np.array(all_val_preds)
    all_val_targets = np.array(all_val_targets)

    aggregate_val_r, aggregate_val_p = pearsonr(all_val_preds, all_val_targets)

    print(f"Aggregate Validation: r={aggregate_val_r:.4f}, p={aggregate_val_p:.2e}")
    print(f"Number of validation samples: {len(all_val_preds)}")

    # Per-fold statistics
    fold_train_r = [f['train_r'] for f in fold_results]
    print(f"Per-fold: Train r={np.mean(fold_train_r):.4f}±{np.std(fold_train_r):.4f}")
    print(f"Per-fold: Nonzero coeffs={np.mean(fold_nonzero_counts):.1f}±{np.std(fold_nonzero_counts):.1f}")

    # Average coefficients across folds
    avg_coefficients = np.mean(fold_coefficients, axis=0)
    std_coefficients = np.std(fold_coefficients, axis=0)

    # Count how often each metric is nonzero across folds
    nonzero_frequency = np.mean([c != 0 for c in fold_coefficients], axis=0)

    results = {
        'aggregate_metrics': {
            'val_r': float(aggregate_val_r),
            'val_p': float(aggregate_val_p),
            'n_val_samples': len(all_val_preds)
        },
        'per_fold_stats': {
            'train_r_mean': float(np.mean(fold_train_r)),
            'train_r_std': float(np.std(fold_train_r)),
            'n_nonzero_mean': float(np.mean(fold_nonzero_counts)),
            'n_nonzero_std': float(np.std(fold_nonzero_counts))
        },
        'average_coefficients': {name: float(coef) for name, coef in zip(metric_names, avg_coefficients)},
        'coefficient_std': {name: float(std) for name, std in zip(metric_names, std_coefficients)},
        'nonzero_frequency': {name: float(freq) for name, freq in zip(metric_names, nonzero_frequency)},
        'fold_results': fold_results,
        'optimization_params': {
            'lambda_l1': lambda_l1,
            'n_iterations': n_iterations,
            'learning_rate': lr,
            'patience': patience,
            'convergence_threshold': convergence_threshold
        }
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='L1-Regularized L2TO Cross-Validation')
    parser.add_argument('--lambda_l1', type=float, default=1.0, help='L1 regularization strength')
    args = parser.parse_args()

    lambda_l1 = args.lambda_l1

    # Configuration
    metrics_path = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results/mergeability/ViT-B-16/pairwise_metrics_N20.json')
    results_base_path = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results/ViT-B-16')
    output_dir = Path(f'/home/ubuntu/thesis/MM/Mergeability-Bench/results/metric_linear_optimization/l2to_cv_l1_lambda{lambda_l1}')
    output_dir.mkdir(parents=True, exist_ok=True)

    merge_methods = ['weight_avg', 'arithmetic', 'tsv', 'isotropic']

    print("="*70)
    print(f"L1-Regularized Linear Optimization with L2TO CV (lambda={lambda_l1})")
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

    print("L2TO: For each pair (A,B), train on pairs where neither task is A or B")
    print(f"Training size per fold: C(18,2) = {18*17//2} pairs")
    print()

    # Run L2TO for each merge method
    all_results = {}

    for method_idx, method in enumerate(merge_methods):
        print("="*70)
        print(f"L2TO Cross-Validation for: {method}")
        print("="*70)
        print()

        # Extract performance for this method
        performance = performance_matrix[:, method_idx]

        # Run L2TO CV
        results = run_l2to_cv(
            metrics_array,
            performance,
            pair_names,
            metric_names,
            lambda_l1=lambda_l1,
            n_iterations=2000,
            lr=0.01,
            patience=100,
            convergence_threshold=1e-5
        )

        all_results[method] = results

        # Save individual method results
        method_output_file = output_dir / f'{method}_l2to_results.json'
        with open(method_output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Saved results to: {method_output_file}")
        print()

    # Save combined results
    combined_output_file = output_dir / 'all_methods_l2to_results.json'
    with open(combined_output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("="*70)
    print("SUMMARY: L2TO Cross-Validation Results")
    print("="*70)
    print()
    print(f"{'Method':<15} {'Val r':<12} {'Train r':<12} {'Nonzero':<12}")
    print("-"*70)
    for method in merge_methods:
        val_r = all_results[method]['aggregate_metrics']['val_r']
        train_r = all_results[method]['per_fold_stats']['train_r_mean']
        n_nonzero = all_results[method]['per_fold_stats']['n_nonzero_mean']
        print(f"{method:<15} {val_r:<12.4f} {train_r:<12.4f} {n_nonzero:<12.1f}")
    print("="*70)

    print()
    print(f"All results saved to: {output_dir}")

    # Show core metrics
    print()
    print("="*70)
    print("Non-zero Coefficients (sorted by frequency across folds)")
    print("="*70)

    for method in merge_methods:
        avg_coefs = all_results[method]['average_coefficients']
        nonzero_freq = all_results[method]['nonzero_frequency']

        # Sort by nonzero frequency
        sorted_items = sorted(nonzero_freq.items(), key=lambda x: x[1], reverse=True)

        print(f"\n{method.upper()} (top 15):")
        print("-" * 70)
        print(f"{'Metric':<45} {'Avg Coef':>10} {'Freq':>8}")
        print("-" * 70)
        for metric, freq in sorted_items[:15]:
            coef = avg_coefs[metric]
            print(f"{metric:<45} {coef:>+10.4f} {freq:>8.0%}")

    print()
    print("="*70)
    print("L2TO Cross-Validation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
