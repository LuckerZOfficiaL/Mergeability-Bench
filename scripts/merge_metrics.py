#!/usr/bin/env python3
"""
Merge two pairwise metrics JSON files into one.
Combines all metrics from both files while preserving metadata.
"""
import json
import sys

def merge_metrics_files(file1_path, file2_path, output_path):
    """
    Merge two metrics JSON files.

    Args:
        file1_path: Path to first metrics file
        file2_path: Path to second metrics file
        output_path: Path for merged output file
    """
    # Read both files
    with open(file1_path, 'r') as f:
        data1 = json.load(f)

    with open(file2_path, 'r') as f:
        data2 = json.load(f)

    # Verify metadata matches
    if data1.get('model_name') != data2.get('model_name'):
        print(f"Warning: model_name differs: {data1.get('model_name')} vs {data2.get('model_name')}")

    if data1.get('datasets') != data2.get('datasets'):
        print("Warning: datasets differ between files")

    # Start with data from first file
    merged_data = data1.copy()

    # Merge metrics from both files
    merged_metrics = data1.get('metrics', {}).copy()
    merged_metrics.update(data2.get('metrics', {}))

    merged_data['metrics'] = merged_metrics

    # Write merged data
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)

    # Print summary
    print(f"Merged metrics files successfully!")
    print(f"Output: {output_path}")
    print(f"\nMetrics from {file1_path}:")
    for metric in data1.get('metrics', {}).keys():
        print(f"  - {metric}")
    print(f"\nMetrics from {file2_path}:")
    for metric in data2.get('metrics', {}).keys():
        print(f"  - {metric}")
    print(f"\nTotal metrics in merged file: {len(merged_metrics)}")

if __name__ == "__main__":
    file1 = "/home/ubuntu/thesis/MM/model-merging/results/mergeability/ViT-B-16/pairwise_metrics_N20.json"
    file2 = "/home/ubuntu/thesis/MM/model-merging/results/mergeability/ViT-B-16/pairwise_metrics_N20_all.json"
    output = "/home/ubuntu/thesis/MM/model-merging/results/mergeability/ViT-B-16/pairwise_metrics_N20_merged.json"

    merge_metrics_files(file1, file2, output)
