"""
Analyze calibration sensitivity for activation and gradient-based mergeability metrics.

This script compares metrics computed with different calibration set sizes
to understand the stability and robustness of these metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# Configuration
CALIBRATION_SIZES = [2, 5, 10, 30, 100, 200]
BASE_DIR = Path("/home/ubuntu/thesis/MM/Mergeability-Bench")
SENSITIVITY_DIR = BASE_DIR / "results/mergeability/ViT-B-16/calibration_sensitivity"
BASELINE_FILE = BASE_DIR / "results/mergeability/ViT-B-16/pairwise_metrics_N20.json"
MERGE_RESULTS_DIR = BASE_DIR / "results/ViT-B-16"
OUTPUT_DIR = BASE_DIR / "results/figs/calibration_sensitivity"

# Merge methods to analyze (without suffix)
MERGE_METHODS = ["arithmetic", "isotropic", "tsv"]

# Metrics that use calibration data
CALIBRATION_METRICS = [
    "activation_l2_distance",
    "activation_cosine_similarity",
    "activation_magnitude_ratio",
    "activation_dot_product",
    "encoder_gradient_cosine_similarity",
    "encoder_gradient_l2_distance",
    "encoder_gradient_dot_product",
    "input_gradient_cosine_similarity",
    "input_gradient_l2_distance",
    "input_gradient_dot_product",
]


def load_mergeability_results(calibration_size: int) -> Dict:
    """Load mergeability results for a given calibration size."""
    if calibration_size == 10:
        filepath = BASELINE_FILE
    else:
        filepath = SENSITIVITY_DIR / str(calibration_size) / "pairwise_metrics_N20.json"

    with open(filepath, "r") as f:
        return json.load(f)


def load_merge_performance(method: str) -> Dict:
    """Load merge performance results for a given method."""
    filepath = MERGE_RESULTS_DIR / method / "all_pairwise_summary_N20.json"
    with open(filepath, "r") as f:
        return json.load(f)


def extract_metric_values(results: Dict, metric_name: str) -> Dict[str, float]:
    """Extract metric values as a dict of pair_key -> value."""
    if metric_name not in results["metrics"]:
        return {}
    return results["metrics"][metric_name].get("pairs", {})


def extract_merge_performance_values(results: Dict, perf_key: str = "acc/test/avg") -> Dict[str, float]:
    """Extract merge performance values as a dict of pair_key -> value."""
    values = {}
    for pair_key, pair_data in results.items():
        if "avg" in pair_data and len(pair_data["avg"]) > 0:
            values[pair_key] = pair_data["avg"][0].get(perf_key, None)
    return values


def compute_ranking_correlation(values1: Dict[str, float], values2: Dict[str, float]) -> Tuple[float, float]:
    """Compute Spearman correlation between two sets of values."""
    common_keys = set(values1.keys()) & set(values2.keys())
    if len(common_keys) < 3:
        return np.nan, np.nan

    v1 = [values1[k] for k in common_keys if values1[k] is not None and values2[k] is not None]
    v2 = [values2[k] for k in common_keys if values1[k] is not None and values2[k] is not None]

    if len(v1) < 3:
        return np.nan, np.nan

    rho, pval = stats.spearmanr(v1, v2)
    return rho, pval


def plot_metric_stability(all_results: Dict[int, Dict], output_dir: Path):
    """Plot metric value stability across calibration sizes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # For each metric, compute mean and std across pairs
    for metric_name in CALIBRATION_METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))

        means = []
        stds = []
        sizes_with_data = []

        for size in CALIBRATION_SIZES:
            if size not in all_results:
                continue
            values = extract_metric_values(all_results[size], metric_name)
            if values:
                valid_values = [v for v in values.values() if v is not None]
                if valid_values:
                    means.append(np.mean(valid_values))
                    stds.append(np.std(valid_values))
                    sizes_with_data.append(size)

        if means:
            ax.errorbar(sizes_with_data, means, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8)
            ax.set_xlabel("Calibration Set Size (per task)", fontsize=12)
            ax.set_ylabel(f"Metric Value", fontsize=12)
            ax.set_title(f"{metric_name}\nMean ± Std across pairs", fontsize=12)
            ax.set_xscale('log')
            ax.set_xticks(CALIBRATION_SIZES)
            ax.set_xticklabels(CALIBRATION_SIZES)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f"stability_{metric_name}.pdf", dpi=150, bbox_inches='tight')
            plt.savefig(output_dir / f"stability_{metric_name}.png", dpi=150, bbox_inches='tight')
            plt.close()

    print(f"Saved stability plots to {output_dir}")


def plot_ranking_correlation_vs_reference(all_results: Dict[int, Dict], reference_size: int, output_dir: Path):
    """Plot ranking correlation vs a reference calibration size."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute correlations for each metric
    correlation_data = []

    for metric_name in CALIBRATION_METRICS:
        ref_values = extract_metric_values(all_results.get(reference_size, {}), metric_name)
        if not ref_values:
            continue

        for size in CALIBRATION_SIZES:
            if size == reference_size or size not in all_results:
                continue

            values = extract_metric_values(all_results[size], metric_name)
            rho, pval = compute_ranking_correlation(ref_values, values)

            correlation_data.append({
                "Metric": metric_name,
                "Calibration Size": size,
                "Spearman ρ": rho,
                "p-value": pval
            })

    if not correlation_data:
        print("No correlation data to plot")
        return

    df = pd.DataFrame(correlation_data)

    # Create heatmap
    pivot_df = df.pivot(index="Metric", columns="Calibration Size", values="Spearman ρ")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="RdYlGn", center=0.9,
                vmin=0.5, vmax=1.0, ax=ax, annot_kws={"size": 9})
    ax.set_title(f"Ranking Correlation (Spearman ρ) vs n={reference_size}", fontsize=14)
    ax.set_xlabel("Calibration Set Size", fontsize=12)
    ax.set_ylabel("Metric", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / f"ranking_correlation_vs_n{reference_size}.pdf", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / f"ranking_correlation_vs_n{reference_size}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Also save as CSV
    pivot_df.to_csv(output_dir / f"ranking_correlation_vs_n{reference_size}.csv")

    print(f"Saved ranking correlation heatmap to {output_dir}")


def plot_pairwise_agreement_heatmap(all_results: Dict[int, Dict], output_dir: Path):
    """Plot pairwise agreement between all calibration sizes for each metric."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric_name in CALIBRATION_METRICS:
        sizes_with_data = [s for s in CALIBRATION_SIZES if s in all_results
                          and extract_metric_values(all_results[s], metric_name)]

        if len(sizes_with_data) < 2:
            continue

        n = len(sizes_with_data)
        corr_matrix = np.ones((n, n))

        for i, size1 in enumerate(sizes_with_data):
            for j, size2 in enumerate(sizes_with_data):
                if i < j:
                    values1 = extract_metric_values(all_results[size1], metric_name)
                    values2 = extract_metric_values(all_results[size2], metric_name)
                    rho, _ = compute_ranking_correlation(values1, values2)
                    corr_matrix[i, j] = rho
                    corr_matrix[j, i] = rho

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="RdYlGn",
                    xticklabels=sizes_with_data, yticklabels=sizes_with_data,
                    vmin=0.5, vmax=1.0, ax=ax)
        ax.set_title(f"{metric_name}\nPairwise Ranking Agreement (Spearman ρ)", fontsize=12)
        ax.set_xlabel("Calibration Size", fontsize=11)
        ax.set_ylabel("Calibration Size", fontsize=11)

        plt.tight_layout()
        plt.savefig(output_dir / f"pairwise_agreement_{metric_name}.pdf", dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / f"pairwise_agreement_{metric_name}.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved pairwise agreement heatmaps to {output_dir}")


def plot_correlation_with_merge_performance(all_results: Dict[int, Dict], output_dir: Path):
    """Plot correlation between metrics and merge performance vs calibration size."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load merge performance for all methods
    merge_perf = {}
    for method in MERGE_METHODS:
        try:
            merge_perf[method] = load_merge_performance(method)
        except FileNotFoundError:
            print(f"Warning: Could not load {method} results")

    if not merge_perf:
        print("No merge performance data found")
        return

    # For each merge method, compute correlation with metrics at each calibration size
    for method, perf_data in merge_perf.items():
        perf_values = extract_merge_performance_values(perf_data)

        if not perf_values:
            continue

        correlation_data = []

        for size in CALIBRATION_SIZES:
            if size not in all_results:
                continue

            for metric_name in CALIBRATION_METRICS:
                metric_values = extract_metric_values(all_results[size], metric_name)
                if not metric_values:
                    continue

                rho, pval = compute_ranking_correlation(metric_values, perf_values)

                # Determine metric type
                if "activation" in metric_name:
                    metric_type = "Activation"
                elif "encoder_gradient" in metric_name:
                    metric_type = "Encoder Gradient"
                else:
                    metric_type = "Input Gradient"

                correlation_data.append({
                    "Metric": metric_name,
                    "Metric Type": metric_type,
                    "Calibration Size": size,
                    "Spearman ρ": rho,
                    "p-value": pval
                })

        if not correlation_data:
            continue

        df = pd.DataFrame(correlation_data)

        # Line plot: correlation vs calibration size, grouped by metric type
        fig, ax = plt.subplots(figsize=(10, 6))

        for metric_name in CALIBRATION_METRICS:
            metric_df = df[df["Metric"] == metric_name]
            if not metric_df.empty:
                ax.plot(metric_df["Calibration Size"], metric_df["Spearman ρ"],
                       marker='o', label=metric_name.replace("_", " "), linewidth=1.5, markersize=5)

        ax.set_xlabel("Calibration Set Size (per task)", fontsize=12)
        ax.set_ylabel("Spearman ρ with Merge Performance", fontsize=12)
        ax.set_title(f"Metric-Performance Correlation vs Calibration Size\n(Merge Method: {method})", fontsize=13)
        ax.set_xscale('log')
        ax.set_xticks(CALIBRATION_SIZES)
        ax.set_xticklabels(CALIBRATION_SIZES)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / f"perf_correlation_{method}.pdf", dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / f"perf_correlation_{method}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Also create a summary heatmap
        pivot_df = df.pivot(index="Metric", columns="Calibration Size", values="Spearman ρ")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    vmin=-0.5, vmax=0.5, ax=ax, annot_kws={"size": 9})
        ax.set_title(f"Correlation with {method} Performance", fontsize=14)
        ax.set_xlabel("Calibration Set Size", fontsize=12)
        ax.set_ylabel("Metric", fontsize=12)

        plt.tight_layout()
        plt.savefig(output_dir / f"perf_correlation_heatmap_{method}.pdf", dpi=150, bbox_inches='tight')
        plt.savefig(output_dir / f"perf_correlation_heatmap_{method}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Save as CSV
        pivot_df.to_csv(output_dir / f"perf_correlation_{method}.csv")

    print(f"Saved performance correlation plots to {output_dir}")


def create_summary_table(all_results: Dict[int, Dict], output_dir: Path):
    """Create a summary table for the paper."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reference: largest calibration size
    reference_size = max(CALIBRATION_SIZES)

    summary_data = []

    for metric_name in CALIBRATION_METRICS:
        ref_values = extract_metric_values(all_results.get(reference_size, {}), metric_name)
        if not ref_values:
            continue

        row = {"Metric": metric_name}

        for size in CALIBRATION_SIZES:
            if size == reference_size:
                row[f"n={size}"] = 1.0
            elif size in all_results:
                values = extract_metric_values(all_results[size], metric_name)
                rho, _ = compute_ranking_correlation(ref_values, values)
                row[f"n={size}"] = rho
            else:
                row[f"n={size}"] = np.nan

        summary_data.append(row)

    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / "summary_table.csv", index=False)

    # Also create a LaTeX version
    latex_str = df.to_latex(index=False, float_format="%.3f", na_rep="-")
    with open(output_dir / "summary_table.tex", "w") as f:
        f.write(latex_str)

    print(f"Saved summary table to {output_dir}")
    print("\nSummary Table (Spearman ρ vs n={reference_size}):")
    print(df.to_string(index=False))


def main():
    """Main analysis function."""
    print("=" * 60)
    print("Calibration Sensitivity Analysis")
    print("=" * 60)

    # Load all results
    print("\nLoading results...")
    all_results = {}

    for size in CALIBRATION_SIZES:
        try:
            all_results[size] = load_mergeability_results(size)
            n_metrics = len([m for m in CALIBRATION_METRICS
                           if m in all_results[size].get("metrics", {})])
            print(f"  n={size}: loaded {n_metrics} calibration metrics")
        except FileNotFoundError as e:
            print(f"  n={size}: not found ({e})")

    if len(all_results) < 2:
        print("Error: Need at least 2 calibration sizes for comparison")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")

    print("\n1. Metric value stability...")
    plot_metric_stability(all_results, OUTPUT_DIR)

    print("\n2. Ranking correlation vs reference (n=200)...")
    plot_ranking_correlation_vs_reference(all_results, 200, OUTPUT_DIR)

    print("\n3. Pairwise agreement heatmaps...")
    plot_pairwise_agreement_heatmap(all_results, OUTPUT_DIR)

    print("\n4. Correlation with merge performance...")
    plot_correlation_with_merge_performance(all_results, OUTPUT_DIR)

    print("\n5. Summary table...")
    create_summary_table(all_results, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
