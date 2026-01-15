#!/usr/bin/env python3
"""
Diagnose the MLP training data to understand why it's not learning.
"""
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_merging.data_loader import (
    load_json, extract_all_mergers_data, normalize_metrics,
    task_level_split
)


def main():
    # Paths
    metrics_path = Path("/home/ubuntu/thesis/MM/model-merging/results/mergeability/ViT-B-16/pairwise_metrics_N20_merged.json")
    results_dir = Path("/home/ubuntu/thesis/MM/model-merging/results/ViT-B-16")

    merge_methods = ['weight_avg', 'arithmetic', 'tsv', 'isotropic']

    print("="*70)
    print("MLP Data Diagnostics")
    print("="*70)
    print()

    # Load metrics data
    print("Loading data...")
    metrics_data = load_json(metrics_path)

    # Load performance data for all merge methods
    performance_data_dict = {}
    for method in merge_methods:
        perf_path = results_dir / method / 'all_pairwise_summary_N20.json'
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

    # Analyze performance distributions
    print("="*70)
    print("PERFORMANCE VALUE DISTRIBUTIONS")
    print("="*70)
    for i, method in enumerate(merge_methods):
        perf = performance_matrix[:, i]
        print(f"\n{method}:")
        print(f"  Mean: {np.mean(perf):.6f}")
        print(f"  Std:  {np.std(perf):.6f}")
        print(f"  Min:  {np.min(perf):.6f}")
        print(f"  Max:  {np.max(perf):.6f}")
        print(f"  Range: {np.max(perf) - np.min(perf):.6f}")

    print()
    print("="*70)
    print("METRICS VALUE DISTRIBUTIONS (before normalization)")
    print("="*70)
    for i, metric_name in enumerate(metric_names[:5]):  # Show first 5
        metric_vals = metrics_array[:, i]
        print(f"\n{metric_name}:")
        print(f"  Mean: {np.mean(metric_vals):.6f}")
        print(f"  Std:  {np.std(metric_vals):.6f}")
        print(f"  Min:  {np.min(metric_vals):.6f}")
        print(f"  Max:  {np.max(metric_vals):.6f}")
    print("\n... (showing first 5 metrics)")
    print()

    # Normalize metrics
    metrics_normalized, min_vals, max_vals = normalize_metrics(metrics_array)

    print("="*70)
    print("NORMALIZED METRICS (should be in [-1, 1])")
    print("="*70)
    print(f"Overall min: {np.min(metrics_normalized):.6f}")
    print(f"Overall max: {np.max(metrics_normalized):.6f}")
    print(f"Overall mean: {np.mean(metrics_normalized):.6f}")
    print(f"Overall std: {np.std(metrics_normalized):.6f}")
    print()

    # Check correlation between performance of different methods
    print("="*70)
    print("CORRELATION BETWEEN MERGE METHODS")
    print("="*70)
    print(f"{'Method 1':<15} {'Method 2':<15} {'Correlation':>12}")
    print("-"*45)
    for i in range(len(merge_methods)):
        for j in range(i+1, len(merge_methods)):
            corr = np.corrcoef(performance_matrix[:, i], performance_matrix[:, j])[0, 1]
            print(f"{merge_methods[i]:<15} {merge_methods[j]:<15} {corr:>12.4f}")
    print()

    # Create visualization
    output_dir = Path("/home/ubuntu/thesis/MM/model-merging/results/learnable_mergeability/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Performance distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, method in enumerate(merge_methods):
        ax = axes[i]
        perf = performance_matrix[:, i]
        ax.hist(perf, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Performance (accuracy)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{method}\nMean={np.mean(perf):.4f}, Std={np.std(perf):.4f}', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'performance_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved performance distributions to: {output_dir / 'performance_distributions.png'}")

    # Plot 2: Correlation matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr_matrix = np.corrcoef(performance_matrix.T)
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(merge_methods)))
    ax.set_yticks(np.arange(len(merge_methods)))
    ax.set_xticklabels(merge_methods, rotation=45, ha='right')
    ax.set_yticklabels(merge_methods)

    # Add correlation values
    for i in range(len(merge_methods)):
        for j in range(len(merge_methods)):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title('Correlation Between Merge Method Performances', fontsize=14)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / 'method_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation matrix to: {output_dir / 'method_correlation_matrix.png'}")

    # Plot 3: Scatter plots between methods
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    plot_idx = 0
    for i in range(len(merge_methods)):
        for j in range(i+1, len(merge_methods)):
            if plot_idx < 6:
                ax = axes[plot_idx]
                ax.scatter(performance_matrix[:, i], performance_matrix[:, j],
                          alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
                corr = np.corrcoef(performance_matrix[:, i], performance_matrix[:, j])[0, 1]
                ax.set_xlabel(f'{merge_methods[i]} performance', fontsize=10)
                ax.set_ylabel(f'{merge_methods[j]} performance', fontsize=10)
                ax.set_title(f'r={corr:.3f}', fontsize=11)
                ax.grid(True, alpha=0.3)
                plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, 6):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'method_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plots to: {output_dir / 'method_scatter_plots.png'}")

    print()
    print("="*70)
    print("Diagnostics complete!")
    print("="*70)


if __name__ == "__main__":
    main()
