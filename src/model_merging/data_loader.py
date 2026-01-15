"""
Data loading utilities for mergeability prediction.
"""
import json
import numpy as np
import torch
from torch.utils.data import Dataset


def load_json(path):
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def extract_all_mergers_data(metrics_data, performance_data_dict, target_metric='acc/test/avg'):
    """
    Extract pairwise metrics and performance for all merge methods.

    Args:
        metrics_data: Dictionary with metrics matrices
        performance_data_dict: Dictionary mapping merge method -> performance data
        target_metric: Performance metric to use as target

    Returns:
        metrics_array: (n_pairs, n_metrics) array of metric values
        performance_dict: Dict mapping merge method -> (n_pairs,) array of performance values
        pair_names: List of pair names
        metric_names: List of metric names
        merge_methods: List of merge method names
    """
    datasets = metrics_data['datasets']
    n_datasets = len(datasets)
    metric_names = list(metrics_data['metrics'].keys())
    merge_methods = list(performance_data_dict.keys())

    # Collect data for all pairs
    pairs_data = []
    pair_names = []

    # Dictionary to store performance for each merge method
    performance_by_method = {method: [] for method in merge_methods}

    for i in range(n_datasets):
        for j in range(n_datasets):
            if i >= j:  # Skip diagonal and lower triangle
                continue

            dataset1 = datasets[i]
            dataset2 = datasets[j]
            pair_key = f"{dataset1}__{dataset2}"

            # Check if this pair exists in ALL performance data
            skip_pair = False
            for method in merge_methods:
                performance_data = performance_data_dict[method]

                if pair_key not in performance_data:
                    skip_pair = True
                    break

                if 'error' in performance_data[pair_key]:
                    skip_pair = True
                    break

                if 'avg' not in performance_data[pair_key]:
                    skip_pair = True
                    break

            if skip_pair:
                continue

            # Extract metrics for this pair
            metric_values = []
            has_missing = False
            for metric_name in metric_names:
                metric_matrix = metrics_data['metrics'][metric_name]['matrix']
                value = metric_matrix[i][j]

                if value is None:
                    has_missing = True
                    break

                metric_values.append(value)

            # Skip pairs with missing metrics
            if has_missing:
                continue

            # Extract performance for all merge methods
            for method in merge_methods:
                performance_data = performance_data_dict[method]
                perf = performance_data[pair_key]['avg'][0][target_metric]
                performance_by_method[method].append(perf)

            pairs_data.append(metric_values)
            pair_names.append(pair_key)

    metrics_array = np.array(pairs_data)

    # Stack performance arrays into a single matrix (n_pairs, n_methods)
    performance_matrix = np.column_stack([
        np.array(performance_by_method[method]) for method in merge_methods
    ])

    return metrics_array, performance_matrix, pair_names, metric_names, merge_methods


def normalize_metrics(metrics_array):
    """
    Normalize each metric to [-1, 1] range using min-max normalization.

    Args:
        metrics_array: (n_pairs, n_metrics) array

    Returns:
        normalized_array: (n_pairs, n_metrics) array normalized to [-1, 1]
        min_vals: Minimum values for each metric
        max_vals: Maximum values for each metric
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


def normalize_performance(performance_matrix):
    """
    Normalize each merge method's performance to [-1, 1] range using min-max normalization.

    Args:
        performance_matrix: (n_pairs, n_methods) array

    Returns:
        normalized_array: (n_pairs, n_methods) array normalized to [-1, 1]
        min_vals: Minimum values for each method
        max_vals: Maximum values for each method
    """
    min_vals = performance_matrix.min(axis=0)
    max_vals = performance_matrix.max(axis=0)

    # Avoid division by zero
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    # Min-max normalization to [0, 1], then scale to [-1, 1]
    normalized = (performance_matrix - min_vals) / ranges
    normalized = normalized * 2 - 1

    return normalized, min_vals, max_vals


def denormalize_performance(normalized_performance, min_vals, max_vals):
    """
    Denormalize performance values from [-1, 1] back to original range.

    Args:
        normalized_performance: (n_pairs, n_methods) array in [-1, 1]
        min_vals: Minimum values for each method
        max_vals: Maximum values for each method

    Returns:
        denormalized_array: (n_pairs, n_methods) array in original range
    """
    # Scale from [-1, 1] to [0, 1]
    scaled = (normalized_performance + 1) / 2

    # Scale from [0, 1] to original range
    ranges = max_vals - min_vals
    denormalized = scaled * ranges + min_vals

    return denormalized


def task_level_split(pair_names, datasets, val_split=0.2, random_seed=42):
    """
    Perform task-level train/validation split.

    Args:
        pair_names: List of pair names (format: "Task1__Task2")
        datasets: List of all dataset/task names
        val_split: Fraction of tasks for validation
        random_seed: Random seed for reproducibility

    Returns:
        train_indices: Array of training pair indices
        val_indices: Array of validation pair indices
        train_tasks: Set of training task names
        val_tasks: Set of validation task names
    """
    n_tasks = len(datasets)
    n_val_tasks = max(1, int(n_tasks * val_split))

    # Randomly select validation tasks
    np.random.seed(random_seed)
    val_task_indices = np.random.choice(n_tasks, n_val_tasks, replace=False)
    val_tasks = set([datasets[i] for i in val_task_indices])
    train_tasks = set([task for task in datasets if task not in val_tasks])

    # Split pairs based on task membership
    train_indices = []
    val_indices = []

    for i, pair_name in enumerate(pair_names):
        task1, task2 = pair_name.split('__')

        # If both tasks are in training set, it's a training pair
        if task1 in train_tasks and task2 in train_tasks:
            train_indices.append(i)
        # Otherwise, it's a validation pair
        else:
            val_indices.append(i)

    return np.array(train_indices), np.array(val_indices), train_tasks, val_tasks


class MergeabilityDataset(Dataset):
    """PyTorch dataset for mergeability prediction."""

    def __init__(self, metrics, performance):
        """
        Initialize dataset.

        Args:
            metrics: (n_pairs, n_metrics) array of normalized metrics
            performance: (n_pairs, n_methods) array of performance values
        """
        self.metrics = torch.FloatTensor(metrics)
        self.performance = torch.FloatTensor(performance)

    def __len__(self):
        return len(self.metrics)

    def __getitem__(self, idx):
        return self.metrics[idx], self.performance[idx]
