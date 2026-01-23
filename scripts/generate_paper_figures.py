#!/usr/bin/env python3
"""Generate figures for the mergeability prediction paper."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path

# Set style with serif fonts (LaTeX-like without requiring LaTeX installation)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
    'mathtext.fontset': 'cm',  # Computer Modern for math
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 25,
    'figure.dpi': 600,
})

# Paths
RESULTS_DIR = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results/metric_linear_optimization/loto_cv_no_leakage')
FIGS_DIR = Path('/home/ubuntu/thesis/MM/Mergeability-Bench/results/figs')
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Method names for display
METHOD_NAMES = {
    'arithmetic': 'Task Arithmetic',
    'weight_avg': 'Weight Averaging',
    'isotropic': 'Isotropic',
    'tsv': 'TSV'
}

# Metric categories
METRIC_CATEGORIES = {
    'Task Vector Geometry': [
        'task_vector_cosine_similarity',
        'task_vector_l2_distance',
        'task_vector_dot_product',
        'weight_space_angle',
        'task_vector_magnitude_ratio',
    ],
    'Effective Rank': [
        'effective_rank',
        'effective_rank_mergeability_score',
        'stable_rank',
        'spectral_gap',
        'singular_value_ratio',
        'layerwise_effective_rank',
        'layerwise_effective_rank_mergeability_score',
    ],
    'Subspace Overlap': [
        'singular_value_overlap',
        'subspace_overlap',
        'right_subspace_overlap_top_k',
        'right_subspace_overlap_bottom_k',
        'interaction_matrix_overlap_top_k',
        'interaction_matrix_overlap_bottom_k',
    ],
    'Activation-Based': [
        'activation_l2_distance',
        'activation_cosine_similarity',
        'activation_magnitude_ratio',
        'activation_dot_product',
    ],
    'Gradient-Based': [
        'encoder_gradient_cosine_similarity',
        'encoder_gradient_l2_distance',
        'encoder_gradient_dot_product',
        'input_gradient_cosine_similarity',
        'input_gradient_l2_distance',
        'input_gradient_dot_product',
    ],
}

# Short metric names for display (LaTeX-safe)
METRIC_SHORT_NAMES = {
    'task_vector_cosine_similarity': r'TV Cosine Sim.',
    'task_vector_l2_distance': r'Task Vector L2 Distance',
    'task_vector_dot_product': r'Task Vector Dot Product',
    'weight_space_angle': r'Task Vector Angle',
    'task_vector_magnitude_ratio': r'Task Vector Magnitude Ratio',
    'effective_rank': r'Effective Rank',
    'effective_rank_mergeability_score': r'Eff Rank Score',
    'stable_rank': r'Stable Rank',
    'spectral_gap': r'Spectral Gap',
    'singular_value_ratio': r'Singular Value Ratio',
    'layerwise_effective_rank': r'Layer Eff. Rank',
    'layerwise_effective_rank_mergeability_score': r'Layer Eff. Rank Score',
    'singular_value_overlap': r'Singular Value Overlap',
    'subspace_overlap': r'Left Subspace Top-$k$',
    'right_subspace_overlap_top_k': r'Right Subspace Top-$k$',
    'right_subspace_overlap_bottom_k': r'Right Subspace Bottom-$k$',
    'interaction_matrix_overlap_top_k': r'Interaction Top-$k$',
    'interaction_matrix_overlap_bottom_k': r'Interaction Bottom-$k$',
    'activation_l2_distance': r'Activation L2 Distance',
    'activation_cosine_similarity': r'Activation Cosine Sim.',
    'activation_magnitude_ratio': r'Activation Magnitude Ratio',
    'activation_dot_product': r'Activation Dot Product',
    'encoder_gradient_cosine_similarity': r'Encoder Gradient Cosine Sim.',
    'encoder_gradient_l2_distance': r'Encoder Gradient L2 Dist.',
    'encoder_gradient_dot_product': r'Encoder Gradient Dot Product',
    'input_gradient_cosine_similarity': r'Input Gradient Cosine Sim.',
    'input_gradient_l2_distance': r'Input Gradient L2 Dist.',
    'input_gradient_dot_product': r'Input Gradient Dot Product',
}


def load_loto_results():
    """Load LOTO results for all methods."""
    results = {}
    for method in ['arithmetic', 'weight_avg', 'isotropic', 'tsv']:
        filepath = RESULTS_DIR / f'{method}_loto_results.json'
        with open(filepath, 'r') as f:
            results[method] = json.load(f)
    return results


def plot_coefficient_heatmap(results):
    """Generate coefficient heatmap across methods."""
    print("Generating coefficient heatmap...")

    methods = ['arithmetic', 'weight_avg', 'isotropic', 'tsv']

    # Get all metrics in category order
    all_metrics = []
    category_boundaries = []
    category_labels = []
    for cat_name, metrics in METRIC_CATEGORIES.items():
        category_boundaries.append(len(all_metrics))
        category_labels.append(cat_name)
        all_metrics.extend(metrics)
    category_boundaries.append(len(all_metrics))

    # Build coefficient matrix
    coef_matrix = np.zeros((len(all_metrics), len(methods)))
    for j, method in enumerate(methods):
        avg_coefs = results[method]['average_coefficients']
        for i, metric in enumerate(all_metrics):
            if metric in avg_coefs:
                coef_matrix[i, j] = avg_coefs[metric]

    # Create figure with extra space on the right for colorbar
    fig, ax = plt.subplots(figsize=(8, 12))

    # Normalize for better visualization (clip extreme values)
    vmax = np.percentile(np.abs(coef_matrix), 95)

    # Plot heatmap
    im = ax.imshow(coef_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-vmax, vmax=vmax)

    # Labels
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_NAMES[m] for m in methods], rotation=45, ha='right', fontsize=18)
    ax.set_yticks(range(len(all_metrics)))
    ax.set_yticklabels([METRIC_SHORT_NAMES.get(m, m) for m in all_metrics], fontsize=18)

    # Add category separators
    for boundary in category_boundaries[1:-1]:
        ax.axhline(y=boundary - 0.5, color='black', linewidth=1.5)

    # Colorbar - position it explicitly to avoid overlap
    cbar_ax = fig.add_axes([0.78, 0.25, 0.03, 0.5])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Average Coefficient', fontsize=18)
    cbar_ax.tick_params(labelsize=16)

    ax.set_xlabel('Merging Method', fontsize=18)
    #ax.set_title('Learned Coefficients Across Merging Methods', fontsize=20)

    # Adjust layout
    plt.subplots_adjust(left=0.25, right=0.75)
    plt.savefig(FIGS_DIR / 'coefficient_heatmap.pdf', bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'coefficient_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved coefficient heatmap to {FIGS_DIR / 'coefficient_heatmap.pdf'}")


def plot_tsv_scatter(results):
    """Generate scatter plot for TSV predictions vs actual."""
    print("Generating TSV scatter plot...")

    tsv_results = results['tsv']
    fold_results = tsv_results['fold_results']

    # Collect all validation predictions and actuals
    all_val_predictions = []
    all_val_actuals = []

    for fold in fold_results:
        if 'val_predictions' in fold and 'val_actuals' in fold:
            all_val_predictions.extend(fold['val_predictions'])
            all_val_actuals.extend(fold['val_actuals'])

    # If predictions not stored, we need to recompute or skip
    if not all_val_predictions:
        print("Warning: Validation predictions not found in results. Using per-fold correlations instead.")
        # Create a summary plot instead
        fig, ax = plt.subplots(figsize=(6, 5))

        val_correlations = [fold['val_r'] for fold in fold_results]
        held_out_tasks = [fold['held_out_task'] for fold in fold_results]

        colors = plt.cm.viridis(np.linspace(0, 1, len(val_correlations)))
        bars = ax.bar(range(len(val_correlations)), val_correlations, color=colors)

        ax.set_xticks(range(len(held_out_tasks)))
        ax.set_xticklabels(held_out_tasks, rotation=90, ha='center', fontsize=7)
        ax.set_ylabel('Validation Correlation ($r$)')
        ax.set_xlabel('Held-Out Task')
        ax.set_title('TSV: Per-Fold Validation Correlations')
        ax.axhline(y=np.mean(val_correlations), color='red', linestyle='--',
                   label='Mean: {:.3f}'.format(np.mean(val_correlations)))
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(FIGS_DIR / 'tsv_validation_by_fold.pdf', bbox_inches='tight')
        plt.savefig(FIGS_DIR / 'tsv_validation_by_fold.png', bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved TSV per-fold plot to {FIGS_DIR / 'tsv_validation_by_fold.pdf'}")
        return

    all_val_predictions = np.array(all_val_predictions)
    all_val_actuals = np.array(all_val_actuals)

    # Compute correlation
    from scipy.stats import pearsonr
    corr, p_value = pearsonr(all_val_predictions, all_val_actuals)

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(all_val_predictions, all_val_actuals, alpha=0.6, s=40,
               edgecolors='k', linewidths=0.5)

    # Best fit line
    z = np.polyfit(all_val_predictions, all_val_actuals, 1)
    p = np.poly1d(z)
    pred_sorted = np.sort(all_val_predictions)
    ax.plot(pred_sorted, p(pred_sorted), 'r-', linewidth=2,
            label=f'Best fit (r={corr:.3f})')

    ax.set_xlabel('Predicted Mergeability Score')
    ax.set_ylabel('Actual Post-Merge Accuracy')
    ax.set_title(f'TSV: Predicted vs Actual Mergeability\n(LOTO Validation, r={corr:.3f}, p={p_value:.2e})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'tsv_scatter.pdf', bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'tsv_scatter.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved TSV scatter plot to {FIGS_DIR / 'tsv_scatter.pdf'}")


def plot_validation_boxplots(results):
    """Generate box plots of per-fold validation correlations."""
    print("Generating validation box plots...")

    methods = ['arithmetic', 'weight_avg', 'isotropic', 'tsv']
    method_labels = [METHOD_NAMES[m] for m in methods]

    # Collect validation correlations per fold
    val_corrs = []
    for method in methods:
        fold_results = results[method]['fold_results']
        corrs = [fold['val_r'] for fold in fold_results]
        val_corrs.append(corrs)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Box plot
    bp = ax.boxplot(val_corrs, labels=method_labels, patch_artist=True)

    # Color boxes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points
    for i, (method, corrs) in enumerate(zip(methods, val_corrs)):
        x = np.random.normal(i + 1, 0.04, size=len(corrs))
        ax.scatter(x, corrs, alpha=0.5, s=20, color='black', zorder=3)

    # Add mean markers
    means = [np.mean(corrs) for corrs in val_corrs]
    ax.scatter(range(1, len(methods) + 1), means, color='red', marker='D',
               s=50, zorder=4, label='Mean')

    ax.set_ylabel('Validation Correlation ($r$)')
    ax.set_xlabel('Merging Method')
    ax.set_title('LOTO Cross-Validation: Per-Fold Validation Correlations')
    ax.legend(loc='lower right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(-0.2, 1.0)

    # Add mean values as text
    for i, mean in enumerate(means):
        ax.text(i + 1, -0.15, '$\\mu$={:.2f}'.format(mean), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'validation_boxplots.pdf', bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'validation_boxplots.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved validation box plots to {FIGS_DIR / 'validation_boxplots.pdf'}")


def plot_category_importance(results):
    """Generate metric category importance bar chart."""
    print("Generating metric category importance chart...")

    methods = ['arithmetic', 'weight_avg', 'isotropic', 'tsv']
    categories = list(METRIC_CATEGORIES.keys())

    # Compute sum of |coefficients| per category per method
    importance = np.zeros((len(methods), len(categories)))

    for j, method in enumerate(methods):
        avg_coefs = results[method]['average_coefficients']
        for i, (cat_name, metrics) in enumerate(METRIC_CATEGORIES.items()):
            cat_importance = sum(abs(avg_coefs.get(m, 0)) for m in metrics)
            importance[j, i] = cat_importance

    # Normalize per method (to show relative importance)
    importance_normalized = importance / importance.sum(axis=1, keepdims=True)

    # Create figure - taller with larger fonts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Absolute importance (stacked bar)
    x = np.arange(len(methods))
    width = 0.6
    bottom = np.zeros(len(methods))
    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))

    for i, (cat, color) in enumerate(zip(categories, colors)):
        ax1.bar(x, importance[:, i], width, bottom=bottom, label=cat, color=color)
        bottom += importance[:, i]

    ax1.set_xticks(x)
    ax1.set_xticklabels([METHOD_NAMES[m] for m in methods], rotation=45, ha='right', fontsize=18)
    ax1.set_ylabel('Sum of $|$Coefficients$|$', fontsize=20)
    ax1.set_title('Absolute Category Importance', fontsize=22)
    ax1.legend(loc='upper left', fontsize=16)
    ax1.tick_params(axis='y', labelsize=18)

    # Plot 2: Relative importance (grouped bar)
    x = np.arange(len(categories))
    width = 0.2

    for i, method in enumerate(methods):
        offset = (i - 1.5) * width
        ax2.bar(x + offset, importance_normalized[i], width,
                label=METHOD_NAMES[method], color=colors[i] if i < len(colors) else f'C{i}')

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right', fontsize=18)
    ax2.set_ylabel('Relative Importance', fontsize=20)
    ax2.set_title('Relative Category Importance per Method', fontsize=22)
    ax2.legend(loc='upper right', fontsize=16)
    ax2.set_ylim(0, 0.5)
    ax2.tick_params(axis='y', labelsize=18)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'category_importance.pdf', bbox_inches='tight')
    plt.savefig(FIGS_DIR / 'category_importance.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved category importance chart to {FIGS_DIR / 'category_importance.pdf'}")


def main():
    print("Loading LOTO results...")
    results = load_loto_results()

    # Generate all figures
    plot_coefficient_heatmap(results)
    plot_tsv_scatter(results)
    plot_validation_boxplots(results)
    plot_category_importance(results)

    print("\nAll figures generated successfully!")
    print(f"Figures saved to: {FIGS_DIR}")


if __name__ == '__main__':
    main()
