#!/usr/bin/env python3
"""Check if all 29 metrics are properly normalized."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model_merging.data_loader import load_json, extract_all_mergers_data
import numpy as np

# Load data
metrics_path = Path('/home/ubuntu/thesis/MM/model-merging/results/mergeability/ViT-B-16/pairwise_metrics_N20.json')
perf_path = Path('/home/ubuntu/thesis/MM/model-merging/results/ViT-B-16/tsv/all_pairwise_summary_N20.json')

metrics_data = load_json(metrics_path)
perf_data = {'tsv': load_json(perf_path)}

# Extract data
metrics_array, performance_matrix, pair_names, metric_names, merge_methods = \
    extract_all_mergers_data(metrics_data, perf_data)

print(f'Metrics array shape: {metrics_array.shape}')
print(f'Number of pairs: {len(pair_names)}')
print(f'Number of metrics: {len(metric_names)}')
print(f'\nMetric names ({len(metric_names)} total):')
for i, name in enumerate(metric_names, 1):
    print(f'{i:2d}. {name}')

# Check raw metric ranges BEFORE normalization
print('\n\nRaw metric ranges (before normalization):')
print(f"{'Metric':<50} {'Min':<12} {'Max':<12} {'Range':<12}")
print('='*90)
for i, name in enumerate(metric_names):
    min_val = metrics_array[:, i].min()
    max_val = metrics_array[:, i].max()
    range_val = max_val - min_val
    print(f'{name:<50} {min_val:<12.6f} {max_val:<12.6f} {range_val:<12.6f}')

# Now normalize
min_vals = metrics_array.min(axis=0)
max_vals = metrics_array.max(axis=0)
ranges = max_vals - min_vals
ranges[ranges == 0] = 1.0
normalized = (metrics_array - min_vals) / ranges
normalized = normalized * 2 - 1

print('\n\nNormalized metric ranges (should all be [-1, 1]):')
print(f"{'Metric':<50} {'Min':<12} {'Max':<12}")
print('='*70)
for i, name in enumerate(metric_names):
    min_val = normalized[:, i].min()
    max_val = normalized[:, i].max()
    status = '✓' if (-1.001 <= min_val <= -0.999 and 0.999 <= max_val <= 1.001) else '✗'
    print(f'{name:<50} {min_val:<12.6f} {max_val:<12.6f} {status}')

# Check if all are in [-1, 1]
all_in_range = np.all((normalized >= -1.001) & (normalized <= 1.001))
print(f'\n\nAll metrics normalized to [-1, 1]: {all_in_range}')
print(f'Overall normalized range: [{normalized.min():.6f}, {normalized.max():.6f}]')
