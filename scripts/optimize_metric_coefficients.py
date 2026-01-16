#!/usr/bin/env python3
"""
Optimize metric coefficients to maximize correlation with post-merge performance.

This script finds the optimal linear combination of mergeability metrics that
best predicts post-merge performance using gradient descent with Adam optimizer.
"""
import json
import numpy as np
import autograd.numpy as anp
from autograd import grad
import argparse
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def load_metrics(metrics_path):
    """Load mergeability metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    return data


def load_performance(performance_path):
    """Load post-merge performance data from JSON file."""
    with open(performance_path, 'r') as f:
        data = json.load(f)
    return data


def extract_pairwise_data(metrics_data, performance_data, target_metric='acc/test/avg'):
    """
    Extract pairwise metrics and performance into aligned arrays.

    Args:
        metrics_data: Dictionary with metrics matrices
        performance_data: Dictionary with post-merge performance
        target_metric: Performance metric to use as target

    Returns:
        metrics_array: (n_pairs, n_metrics) array of metric values
        performance_array: (n_pairs,) array of performance values
        pair_names: List of pair names
        metric_names: List of metric names
    """
    datasets = metrics_data['datasets']
    n_datasets = len(datasets)
    metric_names = list(metrics_data['metrics'].keys())

    # Collect data for all pairs
    pairs_data = []
    pair_names = []

    for i in range(n_datasets):
        for j in range(n_datasets):
            if i >= j:  # Skip diagonal and lower triangle (already covered by upper triangle)
                continue

            dataset1 = datasets[i]
            dataset2 = datasets[j]
            pair_key = f"{dataset1}__{dataset2}"

            # Check if this pair exists in performance data
            if pair_key not in performance_data:
                print(f"Warning: {pair_key} not found in performance data")
                continue

            # Check if this pair has an error
            if 'error' in performance_data[pair_key]:
                print(f"Warning: {pair_key} has error, skipping")
                continue

            # Check if avg key exists
            if 'avg' not in performance_data[pair_key]:
                print(f"Warning: {pair_key} missing 'avg' key, skipping")
                continue

            # Extract performance
            perf = performance_data[pair_key]['avg'][0][target_metric]

            # Extract metrics for this pair
            metric_values = []
            has_missing = False
            for metric_name in metric_names:
                metric_matrix = metrics_data['metrics'][metric_name]['matrix']
                value = metric_matrix[i][j]

                if value is None:
                    print(f"Warning: None value for {pair_key} in metric {metric_name}, skipping pair")
                    has_missing = True
                    break

                metric_values.append(value)

            # Skip pairs with missing metrics
            if has_missing:
                continue

            pairs_data.append(metric_values)
            pair_names.append(pair_key)

    metrics_array = np.array(pairs_data)
    performance_array = np.array([performance_data[pair]['avg'][0][target_metric]
                                   for pair in pair_names])

    return metrics_array, performance_array, pair_names, metric_names


def normalize_metrics(metrics_array):
    """
    Normalize each metric to [-1, 1] range using min-max normalization.
    Min and max are computed across all 190 samples for each metric.

    Args:
        metrics_array: (n_pairs, n_metrics) array

    Returns:
        normalized_array: (n_pairs, n_metrics) array normalized to [-1, 1]
        min_vals: Minimum values for each metric (for denormalization)
        max_vals: Maximum values for each metric (for denormalization)
    """
    min_vals = metrics_array.min(axis=0)
    max_vals = metrics_array.max(axis=0)

    # Avoid division by zero
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    # Min-max normalization to [0, 1], then scale to [-1, 1]
    normalized = (metrics_array - min_vals) / ranges
    normalized = normalized * 2 - 1

    return normalized, min_vals, max_vals


def pearson_correlation_autograd(predictions, targets):
    """
    Compute Pearson correlation coefficient using autograd-compatible operations.

    Args:
        predictions: (n,) array of predicted values
        targets: (n,) array of target values

    Returns:
        correlation: Pearson correlation coefficient
    """
    # Center the data
    pred_mean = anp.mean(predictions)
    target_mean = anp.mean(targets)

    pred_centered = predictions - pred_mean
    target_centered = targets - target_mean

    # Compute correlation
    numerator = anp.sum(pred_centered * target_centered)
    denominator = anp.sqrt(anp.sum(pred_centered**2) * anp.sum(target_centered**2))

    # Add small epsilon to avoid division by zero
    correlation = numerator / (denominator + 1e-8)

    return correlation


def objective_function(coefficients, metrics_normalized, performance):
    """
    Objective function: negative Pearson correlation (we minimize this to maximize correlation).

    Args:
        coefficients: (n_metrics,) array of coefficients
        metrics_normalized: (n_pairs, n_metrics) array of normalized metrics
        performance: (n_pairs,) array of performance values

    Returns:
        negative_correlation: Negative Pearson correlation
    """
    # Compute linear combination
    predictions = anp.dot(metrics_normalized, coefficients)

    # Compute Pearson correlation
    correlation = pearson_correlation_autograd(predictions, performance)

    # Return negative (since we minimize)
    return -correlation


def project_to_simplex(coefficients):
    """
    Project coefficients to sum-to-1 constraint (allows negative values).

    This implements the projection: w' = w - (sum(w) - 1) / n
    """
    n = len(coefficients)
    correction = (np.sum(coefficients) - 1.0) / n
    return coefficients - correction


def optimize_coefficients(metrics_normalized, performance, n_iterations=1000,
                         learning_rate=0.01, print_every=100,
                         convergence_threshold=1e-4, patience=50):
    """
    Optimize coefficients using Adam optimizer with sum-to-1 constraint.

    Constraints:
        - sum(coefficients) = 1 (allows negative coefficients)

    Args:
        metrics_normalized: (n_pairs, n_metrics) array of normalized metrics
        performance: (n_pairs,) array of performance values
        n_iterations: Number of optimization iterations
        learning_rate: Learning rate for Adam
        print_every: Print progress every N iterations
        convergence_threshold: Stop if improvement < threshold for 'patience' iterations
        patience: Number of iterations to wait for improvement before stopping

    Returns:
        best_coefficients: (n_metrics,) array of optimized coefficients (sum=1)
        best_correlation: Best correlation achieved
        history: Dictionary with optimization history
    """
    n_metrics = metrics_normalized.shape[1]

    # Initialize coefficients uniformly
    coefficients = np.ones(n_metrics) / n_metrics

    # Adam optimizer parameters
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    m = np.zeros(n_metrics)  # First moment
    v = np.zeros(n_metrics)  # Second moment

    # Compute gradient function
    grad_fn = grad(objective_function, argnum=0)

    # Track best result
    best_coefficients = coefficients.copy()
    best_correlation = -float('inf')

    # Early stopping tracking
    iterations_without_improvement = 0
    last_improvement_iteration = 0

    # History tracking
    history = {
        'correlation': [],
        'loss': [],
        'coefficient_norm': []
    }

    print("Starting optimization...")
    print(f"Initial correlation: {-objective_function(coefficients, metrics_normalized, performance):.4f}")
    print(f"Convergence threshold: {convergence_threshold}, Patience: {patience}")
    print()

    for iteration in range(n_iterations):
        # Compute gradient
        gradient = grad_fn(coefficients, metrics_normalized, performance)

        # Adam update
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2

        # Bias correction
        m_hat = m / (1 - beta1**(iteration + 1))
        v_hat = v / (1 - beta2**(iteration + 1))

        # Update coefficients
        coefficients = coefficients - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        # Project to sum-to-1 constraint
        coefficients = project_to_simplex(coefficients)

        # Compute current correlation
        current_loss = objective_function(coefficients, metrics_normalized, performance)
        current_correlation = -current_loss

        # Track history
        history['correlation'].append(float(current_correlation))
        history['loss'].append(float(current_loss))
        history['coefficient_norm'].append(float(np.linalg.norm(coefficients)))

        # Update best and check for improvement
        improvement = current_correlation - best_correlation
        if improvement > convergence_threshold:
            best_correlation = current_correlation
            best_coefficients = coefficients.copy()
            iterations_without_improvement = 0
            last_improvement_iteration = iteration
        else:
            iterations_without_improvement += 1

        # Check early stopping
        if iterations_without_improvement >= patience:
            print(f"\nEarly stopping at iteration {iteration + 1}")
            print(f"No improvement > {convergence_threshold} for {patience} iterations")
            print(f"Last improvement at iteration {last_improvement_iteration + 1}")
            break

        # Print progress
        if (iteration + 1) % print_every == 0:
            print(f"Iteration {iteration + 1}/{n_iterations}")
            print(f"  Current correlation: {current_correlation:.4f}")
            print(f"  Best correlation: {best_correlation:.4f}")
            print(f"  Iterations without improvement: {iterations_without_improvement}")
            print(f"  Coefficient sum: {np.sum(coefficients):.6f}")
            print()

    print("Optimization complete!")
    print(f"Final best correlation: {best_correlation:.4f}")
    print(f"Total iterations: {iteration + 1}")
    print()

    # Add stopping info to history
    history['stopped_early'] = iterations_without_improvement >= patience
    history['total_iterations'] = iteration + 1

    return best_coefficients, best_correlation, history


def verify_correlation(coefficients, metrics_normalized, performance):
    """
    Verify the correlation using scipy's pearsonr for validation.
    """
    predictions = np.dot(metrics_normalized, coefficients)
    corr, p_value = pearsonr(predictions, performance)
    return corr, p_value


def plot_scatter(coefficients, metrics_normalized, performance,
                 output_path, split_name, corr, p_value):
    """
    Create scatter plot of predicted vs actual performance.

    Args:
        coefficients: Optimized coefficients
        metrics_normalized: Normalized metrics
        performance: Actual performance values
        output_path: Path to save the plot
        split_name: Name of the split (e.g., 'train', 'validation', 'full')
        corr: Pearson correlation coefficient
        p_value: P-value for correlation
    """
    # Compute predictions
    predictions = np.dot(metrics_normalized, coefficients)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(predictions, performance, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)

    # Add best fit line
    z = np.polyfit(predictions, performance, 1)
    p = np.poly1d(z)
    pred_sorted = np.sort(predictions)
    ax.plot(pred_sorted, p(pred_sorted), 'r-', alpha=0.7, linewidth=2, label=f'Best fit (y={z[0]:.3f}x+{z[1]:.3f})')

    # Labels and title
    ax.set_xlabel('Predicted Mergeability Score (Linear Combination)', fontsize=12)
    ax.set_ylabel('Actual Merge Performance (Accuracy)', fontsize=12)
    ax.set_title(f'{split_name.capitalize()} Set: r={corr:.4f}, p={p_value:.2e}', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved scatter plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Optimize metric coefficients for merge prediction')
    parser.add_argument('--metrics', type=str,
                       default='/home/ubuntu/thesis/MM/model-merging/results/mergeability/ViT-B-16/pairwise_metrics_N20.json',
                       help='Path to merged metrics file')
    parser.add_argument('--performance', type=str,
                       default='/home/ubuntu/thesis/MM/model-merging/results/ViT-B-16/arithmetic/all_pairwise_summary_N20.json',
                       help='Path to performance file')
    parser.add_argument('--output', type=str,
                       default='/home/ubuntu/thesis/MM/model-merging/results/metric_linear_optimization/arithmetic_pearson.json',
                       help='Output path for optimized coefficients')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of optimization iterations')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                       help='Learning rate for Adam optimizer')
    parser.add_argument('--convergence-threshold', type=float, default=1e-4,
                       help='Convergence threshold for early stopping')
    parser.add_argument('--patience', type=int, default=50,
                       help='Number of iterations without improvement before stopping')
    parser.add_argument('--target-metric', type=str, default='acc/test/avg',
                       help='Target performance metric to correlate with')
    parser.add_argument('--validation-split', type=float, default=0.2,
                       help='Fraction of pairs to use for validation (0.0 to 1.0)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducible train/val splitting')

    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create figs directory within the metric_linear_optimization results folder
    figs_dir = output_path.parent / 'figs'
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Determine merge method from output path
    merge_method = output_path.stem.split('_')[0]  # e.g., 'arithmetic' from 'arithmetic_pearson'

    # Load data
    print("Loading data...")
    metrics_data = load_metrics(args.metrics)
    performance_data = load_performance(args.performance)

    # Extract pairwise data
    print("Extracting pairwise data...")
    metrics_array, performance_array, pair_names, metric_names = extract_pairwise_data(
        metrics_data, performance_data, args.target_metric
    )

    print(f"Number of pairs: {len(pair_names)}")
    print(f"Number of metrics: {len(metric_names)}")
    print(f"Metrics: {metric_names}")
    print()

    # Normalize metrics (before splitting to use same normalization for train and val)
    print("Normalizing metrics...")
    metrics_normalized, min_vals, max_vals = normalize_metrics(metrics_array)
    print("Normalization complete.")
    print()

    # Perform train/validation split if validation_split > 0
    if args.validation_split > 0.0:
        print(f"Performing task-level train/validation split (validation={args.validation_split:.1%}, seed={args.random_seed})...")

        # Get list of unique tasks from the dataset
        all_tasks = metrics_data['datasets']
        n_tasks = len(all_tasks)

        # Determine number of validation tasks based on validation_split
        n_val_tasks = max(1, int(n_tasks * args.validation_split))
        n_train_tasks = n_tasks - n_val_tasks

        # Randomly select validation tasks
        np.random.seed(args.random_seed)
        val_task_indices = np.random.choice(n_tasks, n_val_tasks, replace=False)
        val_tasks = set([all_tasks[i] for i in val_task_indices])
        train_tasks = set([task for task in all_tasks if task not in val_tasks])

        print(f"  Total tasks: {n_tasks}")
        print(f"  Training tasks ({n_train_tasks}): {sorted(train_tasks)}")
        print(f"  Validation tasks ({n_val_tasks}): {sorted(val_tasks)}")
        print()

        # Split pairs based on task membership
        # Training: both tasks in train_tasks
        # Validation: at least one task in val_tasks
        train_indices = []
        val_indices = []

        for i, pair_name in enumerate(pair_names):
            # Parse pair name (format: "Task1__Task2")
            task1, task2 = pair_name.split('__')

            # If both tasks are in training set, it's a training pair
            if task1 in train_tasks and task2 in train_tasks:
                train_indices.append(i)
            # Otherwise, it's a validation pair (at least one task is in validation)
            else:
                val_indices.append(i)

        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)

        metrics_train = metrics_normalized[train_indices]
        performance_train = performance_array[train_indices]
        metrics_val = metrics_normalized[val_indices]
        performance_val = performance_array[val_indices]

        train_pair_names = [pair_names[i] for i in train_indices]
        val_pair_names = [pair_names[i] for i in val_indices]

        print(f"  Training pairs: {len(train_pair_names)} (pairs among {n_train_tasks} training tasks)")
        print(f"  Validation pairs: {len(val_pair_names)} (pairs involving {n_val_tasks} validation tasks)")
        print()

        # Optimize on training set
        print("Optimizing on training set...")
        best_coefficients, best_correlation_train, history = optimize_coefficients(
            metrics_train, performance_train,
            n_iterations=args.iterations,
            learning_rate=args.learning_rate,
            convergence_threshold=args.convergence_threshold,
            patience=args.patience
        )

        # Evaluate on training set
        train_corr, train_p_value = verify_correlation(best_coefficients, metrics_train, performance_train)
        print(f"Training correlation (scipy): {train_corr:.4f} (p-value: {train_p_value:.2e})")
        print()

        # Evaluate on validation set
        print("Evaluating on validation set...")
        val_corr, val_p_value = verify_correlation(best_coefficients, metrics_val, performance_val)
        print(f"Validation correlation (scipy): {val_corr:.4f} (p-value: {val_p_value:.2e})")
        print()

        # Evaluate on full dataset for reference
        full_corr, full_p_value = verify_correlation(best_coefficients, metrics_normalized, performance_array)
        print(f"Full dataset correlation (scipy): {full_corr:.4f} (p-value: {full_p_value:.2e})")
        print()

        # Generate scatter plots
        print("Generating scatter plots...")
        plot_scatter(best_coefficients, metrics_train, performance_train,
                    figs_dir / f'{merge_method}_train_scatter.png',
                    'train', train_corr, train_p_value)
        plot_scatter(best_coefficients, metrics_val, performance_val,
                    figs_dir / f'{merge_method}_validation_scatter.png',
                    'validation', val_corr, val_p_value)
        plot_scatter(best_coefficients, metrics_normalized, performance_array,
                    figs_dir / f'{merge_method}_full_scatter.png',
                    'full', full_corr, full_p_value)
        print()

    else:
        # No validation split - use all data for training
        print("No validation split - using all pairs for optimization...")
        train_pair_names = pair_names
        val_pair_names = []

        best_coefficients, best_correlation_train, history = optimize_coefficients(
            metrics_normalized, performance_array,
            n_iterations=args.iterations,
            learning_rate=args.learning_rate,
            convergence_threshold=args.convergence_threshold,
            patience=args.patience
        )

        train_corr, train_p_value = verify_correlation(best_coefficients, metrics_normalized, performance_array)
        print(f"Training correlation (scipy): {train_corr:.4f} (p-value: {train_p_value:.2e})")
        print()

        val_corr = None
        val_p_value = None
        full_corr = train_corr
        full_p_value = train_p_value

        # Generate scatter plot
        print("Generating scatter plot...")
        plot_scatter(best_coefficients, metrics_normalized, performance_array,
                    figs_dir / f'{merge_method}_full_scatter.png',
                    'full', full_corr, full_p_value)
        print()

    # Print results
    print("=" * 60)
    print("OPTIMIZED COEFFICIENTS")
    print("=" * 60)
    print(f"{'Metric':<45} {'Coefficient':>12}")
    print("-" * 60)
    for name, coef in zip(metric_names, best_coefficients):
        print(f"{name:<45} {coef:>12.6f}")
    print("-" * 60)
    print(f"{'Sum':<45} {np.sum(best_coefficients):>12.6f}")
    print("=" * 60)
    print()

    # Save results
    results = {
        'coefficients': {name: float(coef) for name, coef in zip(metric_names, best_coefficients)},
        'correlation': {
            'train': float(train_corr),
            'train_p_value': float(train_p_value),
        },
        'n_pairs': {
            'total': len(pair_names),
            'train': len(train_pair_names),
            'validation': len(val_pair_names) if args.validation_split > 0 else 0
        },
        'n_metrics': len(metric_names),
        'target_metric': args.target_metric,
        'optimization': {
            'n_iterations': args.iterations,
            'total_iterations': history['total_iterations'],
            'stopped_early': history['stopped_early'],
            'learning_rate': args.learning_rate,
            'convergence_threshold': args.convergence_threshold,
            'patience': args.patience,
            'final_loss': float(history['loss'][-1]),
            'validation_split': args.validation_split,
            'random_seed': args.random_seed
        },
        'normalization': {
            'min_vals': {name: float(val) for name, val in zip(metric_names, min_vals)},
            'max_vals': {name: float(val) for name, val in zip(metric_names, max_vals)}
        }
    }

    # Add validation results if split was performed
    if args.validation_split > 0.0:
        results['correlation']['validation'] = float(val_corr)
        results['correlation']['validation_p_value'] = float(val_p_value)
        results['correlation']['full_dataset'] = float(full_corr)
        results['correlation']['full_dataset_p_value'] = float(full_p_value)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {args.output}")
    print()


if __name__ == "__main__":
    main()
