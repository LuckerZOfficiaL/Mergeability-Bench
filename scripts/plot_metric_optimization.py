#!/usr/bin/env python3
"""
Generate scatter plots of optimized linear combinations vs actual performance.

Creates scatter plots with regression lines showing the correlation between
the optimized metric linear combination and post-merge performance.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, linregress
import argparse


def load_optimization_results(result_path):
    """Load optimization results from JSON file."""
    with open(result_path, 'r') as f:
        data = json.load(f)
    return data


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


def extract_pairwise_data(metrics_data, performance_data, coefficients_dict,
                          normalization, target_metric='acc/test/avg'):
    """
    Extract pairwise metrics, compute linear combination, and get actual performance.

    Args:
        metrics_data: Dictionary with metrics matrices
        performance_data: Dictionary with post-merge performance
        coefficients_dict: Dictionary mapping metric names to coefficients
        normalization: Dictionary with min_vals and max_vals
        target_metric: Performance metric to use as target

    Returns:
        predictions: Array of linear combination values
        actual: Array of actual performance values
        pair_names: List of pair names
    """
    datasets = metrics_data['datasets']
    n_datasets = len(datasets)
    metric_names = list(coefficients_dict.keys())

    min_vals = np.array([normalization['min_vals'][m] for m in metric_names])
    max_vals = np.array([normalization['max_vals'][m] for m in metric_names])
    coefficients = np.array([coefficients_dict[m] for m in metric_names])

    predictions = []
    actual = []
    pair_names = []

    for i in range(n_datasets):
        for j in range(n_datasets):
            if i >= j:  # Skip diagonal and lower triangle
                continue

            dataset1 = datasets[i]
            dataset2 = datasets[j]
            pair_key = f"{dataset1}__{dataset2}"

            # Check if this pair exists in performance data
            if pair_key not in performance_data:
                continue

            # Check if this pair has an error
            if 'error' in performance_data[pair_key]:
                continue

            # Check if avg key exists
            if 'avg' not in performance_data[pair_key]:
                continue

            # Extract performance
            perf = performance_data[pair_key]['avg'][0][target_metric]

            # Extract and normalize metrics for this pair
            metric_values = []
            for metric_name in metric_names:
                metric_matrix = metrics_data['metrics'][metric_name]['matrix']
                value = metric_matrix[i][j]
                if value is None:
                    value = 0.0
                metric_values.append(value)

            metric_values = np.array(metric_values)

            # Normalize metrics to [-1, 1]
            ranges = max_vals - min_vals
            ranges[ranges == 0] = 1.0
            normalized = (metric_values - min_vals) / ranges
            normalized = normalized * 2 - 1

            # Compute linear combination
            prediction = np.dot(normalized, coefficients)

            predictions.append(prediction)
            actual.append(perf)
            pair_names.append(pair_key)

    return np.array(predictions), np.array(actual), pair_names


def create_scatter_plot(predictions, actual, correlation, p_value, method_name, output_path):
    """
    Create and save scatter plot with regression line.

    Args:
        predictions: Array of predicted values (linear combination)
        actual: Array of actual performance values
        correlation: Pearson correlation coefficient
        p_value: p-value for correlation
        method_name: Name of merging method
        output_path: Path to save the plot
    """
    # Compute linear regression
    slope, intercept, r_value, p_val, std_err = linregress(predictions, actual)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot
    ax.scatter(predictions, actual, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

    # Regression line
    x_line = np.array([predictions.min(), predictions.max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'Regression line')

    # Add correlation info to plot
    ax.text(0.05, 0.95,
            f'r = {correlation:.4f}\np < {p_value:.2e}\n$R^2$ = {correlation**2:.4f}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Labels and title
    ax.set_xlabel('Optimized Linear Combination of Metrics', fontsize=12)
    ax.set_ylabel('Actual Post-Merge Performance (acc/test/avg)', fontsize=12)
    ax.set_title(f'Mergeability Prediction: {method_name.upper()}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate scatter plots for metric optimization results')
    parser.add_argument('--metrics', type=str,
                       default='/home/ubuntu/thesis/MM/model-merging/results/mergeability/ViT-B-16/pairwise_metrics_N20_merged.json',
                       help='Path to merged metrics file')
    parser.add_argument('--results-dir', type=str,
                       default='/home/ubuntu/thesis/MM/model-merging/results/metric_linear_optimization',
                       help='Directory containing optimization results')
    parser.add_argument('--output-dir', type=str,
                       default='/home/ubuntu/thesis/MM/model-merging/results/metric_linear_optimization/figs',
                       help='Output directory for plots')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['arithmetic', 'weight_avg', 'isotropic', 'tsv'],
                       help='Merging methods to plot')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics data once
    print("Loading metrics data...")
    metrics_data = load_metrics(args.metrics)

    # Process each method
    for method in args.methods:
        print(f"\nProcessing {method}...")

        # Paths
        result_path = Path(args.results_dir) / f"{method}_pearson.json"
        performance_path = Path("/home/ubuntu/thesis/MM/model-merging/results/ViT-B-16") / method / "all_pairwise_summary_N20.json"
        output_path = output_dir / f"{method}_pearson.jpg"

        # Check if result file exists
        if not result_path.exists():
            print(f"Warning: Result file not found: {result_path}")
            continue

        # Check if performance file exists
        if not performance_path.exists():
            print(f"Warning: Performance file not found: {performance_path}")
            continue

        # Load data
        print(f"  Loading optimization results...")
        results = load_optimization_results(result_path)

        print(f"  Loading performance data...")
        performance_data = load_performance(performance_path)

        # Extract data and compute predictions
        print(f"  Computing predictions...")
        predictions, actual, pair_names = extract_pairwise_data(
            metrics_data,
            performance_data,
            results['coefficients'],
            results['normalization']
        )

        print(f"  Number of pairs: {len(pair_names)}")
        print(f"  Correlation: {results['verified_correlation']:.4f}")

        # Create plot
        print(f"  Creating plot...")
        create_scatter_plot(
            predictions,
            actual,
            results['verified_correlation'],
            results['p_value'],
            method,
            output_path
        )

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
