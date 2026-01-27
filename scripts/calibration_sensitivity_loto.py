#!/usr/bin/env python3
"""
Calibration Set Sensitivity Analysis for Linear Mergeability Prediction.

This script evaluates how the number of calibration samples affects
the predictive performance of learned linear coefficients.

For each calibration set size N, it:
1. Loads metrics and performance data computed with N samples per task
2. Runs LOTO cross-validation (without L1 regularization)
3. Saves results to separate folders

Usage:
    python scripts/calibration_sensitivity_loto.py --n_samples 2 8 20
    python scripts/calibration_sensitivity_loto.py --n_samples 1 5 10 30 100  # if data exists
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


def linear_optimization_single_fold(metrics_train, performance_train,
                                     metrics_val, performance_val,
                                     n_iterations=1000, lr=0.01,
                                     patience=50, convergence_threshold=1e-4):
    """
    Optimize linear coefficients for a single fold (no L1 regularization).

    Returns:
        coefficients: Optimized coefficients
        train_r: Training Pearson r
        val_r: Validation Pearson r
        n_iters: Number of iterations run
    """
    n_metrics = metrics_train.shape[1]

    # Initialize coefficients
    coefficients = torch.randn(n_metrics, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([coefficients], lr=lr)

    # Convert to tensors
    X_train = torch.FloatTensor(metrics_train)
    y_train = torch.FloatTensor(performance_train)
    X_val = torch.FloatTensor(metrics_val)
    y_val = torch.FloatTensor(performance_val)

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
        loss = -correlation  # Maximize correlation = minimize negative correlation

        # Backward pass
        loss.backward()

        # Constraint: coefficients sum to 1
        with torch.no_grad():
            coefficients.data = coefficients.data / (coefficients.data.sum() + 1e-8)

        optimizer.step()

        # Early stopping
        if loss < best_train_loss - convergence_threshold:
            best_train_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Final evaluation
    with torch.no_grad():
        train_pred = (X_train @ coefficients).numpy()
        val_pred = (X_val @ coefficients).numpy()

    train_r, _ = pearsonr(train_pred, performance_train)
    val_r, _ = pearsonr(val_pred, performance_val)

    return coefficients.detach().numpy(), train_r, val_r, iteration + 1


def run_loto_cv(metrics_array, performance_array, pair_names, all_tasks,
                metric_names, n_iterations=1000, lr=0.01,
                patience=50, convergence_threshold=1e-4):
    """
    Run Leave-One-Task-Out cross-validation.

    Returns:
        results: Dictionary with fold results and aggregate metrics
    """
    n_tasks = len(all_tasks)

    # Storage for results
    fold_results = []
    all_train_preds = []
    all_train_targets = []
    all_val_preds = []
    all_val_targets = []
    fold_coefficients = []

    for fold_idx, held_out_task in enumerate(all_tasks):
        # Determine train and validation tasks
        train_tasks = [t for t in all_tasks if t != held_out_task]

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

        if len(val_indices) == 0:
            continue

        # Extract raw data for this fold
        metrics_train_raw = metrics_array[train_indices]
        performance_train = performance_array[train_indices]
        metrics_val_raw = metrics_array[val_indices]
        performance_val = performance_array[val_indices]

        # Normalize using ONLY training data statistics (no leakage)
        metrics_train, min_vals, max_vals = normalize_metrics(metrics_train_raw)
        metrics_val = normalize_metrics_with_stats(metrics_val_raw, min_vals, max_vals)

        # Optimize coefficients for this fold
        coefficients, train_r, val_r, n_iters = linear_optimization_single_fold(
            metrics_train, performance_train,
            metrics_val, performance_val,
            n_iterations=n_iterations,
            lr=lr,
            patience=patience,
            convergence_threshold=convergence_threshold
        )

        # Store predictions for aggregate evaluation
        train_preds = metrics_train @ coefficients
        val_preds = metrics_val @ coefficients

        all_train_preds.append(train_preds)
        all_train_targets.append(performance_train)
        all_val_preds.append(val_preds)
        all_val_targets.append(performance_val)
        fold_coefficients.append(coefficients)

        # Store fold results
        fold_results.append({
            'fold': fold_idx,
            'held_out_task': held_out_task,
            'n_train_pairs': len(train_indices),
            'n_val_pairs': len(val_indices),
            'train_r': float(train_r),
            'val_r': float(val_r),
            'n_iterations': int(n_iters),
            'coefficients': {name: float(coef) for name, coef in zip(metric_names, coefficients)}
        })

    # Aggregate results
    all_train_preds = np.concatenate(all_train_preds)
    all_train_targets = np.concatenate(all_train_targets)
    all_val_preds = np.concatenate(all_val_preds)
    all_val_targets = np.concatenate(all_val_targets)

    # Compute aggregate correlations
    aggregate_train_r, aggregate_train_p = pearsonr(all_train_preds, all_train_targets)
    aggregate_val_r, aggregate_val_p = pearsonr(all_val_preds, all_val_targets)

    # Per-fold statistics
    fold_train_r = [f['train_r'] for f in fold_results]
    fold_val_r = [f['val_r'] for f in fold_results]

    # Average coefficients across folds
    avg_coefficients = np.mean(fold_coefficients, axis=0)
    std_coefficients = np.std(fold_coefficients, axis=0)

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
            'val_r_std': float(np.std(fold_val_r))
        },
        'average_coefficients': {name: float(coef) for name, coef in zip(metric_names, avg_coefficients)},
        'coefficient_std': {name: float(std) for name, std in zip(metric_names, std_coefficients)},
        'fold_results': fold_results,
        'optimization_params': {
            'n_iterations': n_iterations,
            'learning_rate': lr,
            'patience': patience,
            'convergence_threshold': convergence_threshold
        }
    }

    return results


def run_for_n_samples(n_samples, base_results_path, output_base_dir, merge_methods, use_calibration_suffix=True):
    """Run LOTO CV for a specific calibration set size.

    Args:
        n_samples: Number of calibration samples
        base_results_path: Base path for results
        output_base_dir: Output directory for results
        merge_methods: List of merge methods to evaluate
        use_calibration_suffix: If True, look for files with _calibration suffix
    """
    results_base_path = base_results_path / 'ViT-B-16'
    output_dir = output_base_dir / f'calibration_N{n_samples}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try different file naming patterns
    if use_calibration_suffix:
        metrics_path = base_results_path / 'mergeability' / 'ViT-B-16' / f'pairwise_metrics_N{n_samples}_calibration.json'
        perf_suffix = f'N{n_samples}_calibration'
    else:
        metrics_path = base_results_path / 'mergeability' / 'ViT-B-16' / f'pairwise_metrics_N{n_samples}.json'
        perf_suffix = f'N{n_samples}'

    # Check if metrics file exists
    if not metrics_path.exists():
        # Try alternative naming
        alt_path = base_results_path / 'mergeability' / 'ViT-B-16' / f'pairwise_metrics_N{n_samples}.json'
        if alt_path.exists():
            metrics_path = alt_path
            perf_suffix = f'N{n_samples}'
        else:
            print(f"WARNING: Metrics file not found: {metrics_path}")
            print(f"         Skipping N={n_samples}")
            return None

    print("=" * 70)
    print(f"Running LOTO CV for Calibration Set Size N={n_samples}")
    print("=" * 70)
    print()

    # Load metrics data
    print("Loading data...")
    metrics_data = load_json(metrics_path)

    # Load performance data for all methods
    performance_data_dict = {}
    for method in merge_methods:
        perf_path = results_base_path / method / f'all_pairwise_summary_{perf_suffix}.json'
        if not perf_path.exists():
            # Try without calibration suffix
            alt_perf_path = results_base_path / method / f'all_pairwise_summary_N{n_samples}.json'
            if alt_perf_path.exists():
                perf_path = alt_perf_path
            else:
                print(f"  Warning: {perf_path} not found, skipping {method}")
                continue
        performance_data_dict[method] = load_json(perf_path)

    if not performance_data_dict:
        print(f"  No performance data found for N={n_samples}, skipping")
        return None

    print(f"Loaded data for methods: {list(performance_data_dict.keys())}")

    # Extract pairwise data
    metrics_array, performance_matrix, pair_names, metric_names, available_methods = \
        extract_all_mergers_data(metrics_data, performance_data_dict)

    print(f"Number of pairs: {len(pair_names)}")
    print(f"Number of metrics: {len(metric_names)}")
    print()

    # Get list of tasks
    all_tasks = metrics_data['datasets']

    # Run LOTO for each merge method
    all_results = {}

    for method_idx, method in enumerate(available_methods):
        print(f"  {method}...", end=" ", flush=True)

        # Extract performance for this method
        performance = performance_matrix[:, method_idx]

        # Run LOTO CV
        results = run_loto_cv(
            metrics_array,
            performance,
            pair_names,
            all_tasks,
            metric_names,
            n_iterations=1000,
            lr=0.01,
            patience=50,
            convergence_threshold=1e-4
        )

        all_results[method] = results

        val_r = results['aggregate_metrics']['val_r']
        print(f"val_r = {val_r:.4f}")

        # Save individual method results
        method_output_file = output_dir / f'{method}_loto_results.json'
        with open(method_output_file, 'w') as f:
            json.dump(results, f, indent=2)

    # Save combined results with metadata
    combined_results = {
        'n_calibration_samples': n_samples,
        'methods': all_results
    }

    combined_output_file = output_dir / 'all_methods_loto_results.json'
    with open(combined_output_file, 'w') as f:
        json.dump(combined_results, f, indent=2)

    print(f"Results saved to: {output_dir}")
    print()

    return all_results


def compare_results(output_base_dir, n_samples_list, reference_n=20):
    """Compare results across different calibration set sizes."""

    print()
    print("=" * 70)
    print("CALIBRATION SET SENSITIVITY ANALYSIS")
    print("=" * 70)
    print()

    # Load all results
    all_n_results = {}
    available_methods = None

    for n in n_samples_list:
        results_file = output_base_dir / f'calibration_N{n}' / 'all_methods_loto_results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                all_n_results[n] = data['methods']
                if available_methods is None:
                    available_methods = list(data['methods'].keys())
        else:
            print(f"  Results not found for N={n}")

    if not all_n_results:
        print("No results found to compare!")
        return

    # Sort N values
    sorted_n = sorted(all_n_results.keys())

    # Print comparison table
    print("Aggregate Validation Pearson Correlation (r) by Calibration Set Size:")
    print()

    # Header
    header = f"{'Method':<15}" + "".join(f"{'N='+str(n):>10}" for n in sorted_n)
    print(header)
    print("-" * len(header))

    # Data rows
    for method in available_methods:
        row = f"{method:<15}"
        for n in sorted_n:
            if n in all_n_results and method in all_n_results[n]:
                val_r = all_n_results[n][method]['aggregate_metrics']['val_r']
                row += f"{val_r:>10.4f}"
            else:
                row += f"{'N/A':>10}"
        print(row)

    print()

    # Compute sensitivity metrics relative to reference
    if reference_n in all_n_results:
        print(f"Relative Change vs Reference (N={reference_n}):")
        print()

        header = f"{'Method':<15}" + "".join(f"{'N='+str(n):>10}" for n in sorted_n if n != reference_n)
        print(header)
        print("-" * len(header))

        for method in available_methods:
            if method not in all_n_results[reference_n]:
                continue
            ref_val = all_n_results[reference_n][method]['aggregate_metrics']['val_r']
            row = f"{method:<15}"
            for n in sorted_n:
                if n == reference_n:
                    continue
                if n in all_n_results and method in all_n_results[n]:
                    val_r = all_n_results[n][method]['aggregate_metrics']['val_r']
                    if ref_val != 0:
                        pct_change = (val_r - ref_val) / abs(ref_val) * 100
                        row += f"{pct_change:>+9.1f}%"
                    else:
                        row += f"{'N/A':>10}"
                else:
                    row += f"{'N/A':>10}"
            print(row)

    print()

    # Per-fold standard deviation comparison
    print("Per-Fold Validation Correlation Std Dev (stability measure):")
    print()

    header = f"{'Method':<15}" + "".join(f"{'N='+str(n):>10}" for n in sorted_n)
    print(header)
    print("-" * len(header))

    for method in available_methods:
        row = f"{method:<15}"
        for n in sorted_n:
            if n in all_n_results and method in all_n_results[n]:
                val_r_std = all_n_results[n][method]['per_fold_stats']['val_r_std']
                row += f"{val_r_std:>10.4f}"
            else:
                row += f"{'N/A':>10}"
        print(row)

    print()

    # Save comparison summary
    summary = {
        'n_samples_tested': sorted_n,
        'reference_n': reference_n,
        'methods': available_methods,
        'validation_r': {},
        'validation_r_std': {},
    }

    for method in available_methods:
        summary['validation_r'][method] = {}
        summary['validation_r_std'][method] = {}
        for n in sorted_n:
            if n in all_n_results and method in all_n_results[n]:
                summary['validation_r'][method][str(n)] = all_n_results[n][method]['aggregate_metrics']['val_r']
                summary['validation_r_std'][method][str(n)] = all_n_results[n][method]['per_fold_stats']['val_r_std']

    summary_file = output_base_dir / 'calibration_sensitivity_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description='Calibration Set Sensitivity Analysis')
    parser.add_argument('--n_samples', type=int, nargs='+', default=[1, 5, 10, 30, 100],
                        help='List of calibration set sizes to test (default: 1 5 10 30 100)')
    parser.add_argument('--reference_n', type=int, default=10,
                        help='Reference N for comparison (default: 10, the standard setting)')
    parser.add_argument('--compare_only', action='store_true',
                        help='Only run comparison on existing results')
    parser.add_argument('--use_default_data', action='store_true',
                        help='Use the default N=20 data file (baseline run)')
    args = parser.parse_args()

    # Configuration
    base_results_path = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results')
    output_base_dir = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results/metric_linear_optimization/calibration_sensitivity')
    output_base_dir.mkdir(parents=True, exist_ok=True)

    merge_methods = ['weight_avg', 'arithmetic', 'tsv', 'isotropic']

    if args.use_default_data:
        # Run with just the default N=20 data as baseline
        print("Using default N=20 data file as baseline...")
        run_for_n_samples(20, base_results_path, output_base_dir, merge_methods, use_calibration_suffix=False)
        args.n_samples = [20]
        args.reference_n = 20
    elif not args.compare_only:
        # Run LOTO for each calibration set size
        for n in args.n_samples:
            run_for_n_samples(n, base_results_path, output_base_dir, merge_methods)

    # Compare results
    compare_results(output_base_dir, args.n_samples, args.reference_n)

    print()
    print("=" * 70)
    print("Calibration Sensitivity Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
