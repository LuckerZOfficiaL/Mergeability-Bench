"""
Compute mergeability metrics between fine-tuned models.

This script computes various metrics to predict model merging outcomes
without actually performing the merge. It computes pairwise metrics
between all datasets, returning an NxN triangular matrix structure.

Usage:
    python scripts/compute_mergeability.py mergeability.metrics=[task_vector_cosine_similarity]
    python scripts/compute_mergeability.py 'mergeability.datasets=[CIFAR10,DTD,MNIST]'
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import hydra
import omegaconf
import torch
from omegaconf import DictConfig

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything

# Force the execution of __init__.py if this file is executed directly.
import model_merging  # noqa
from model_merging.metrics import (
    METRIC_REGISTRY,
    compute_metric,
    compute_all_metrics,
)
from model_merging.utils.io_utils import load_model_from_hf
from model_merging.utils.utils import compute_task_dict, print_memory

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def run(cfg: DictConfig) -> Dict:
    """Compute pairwise mergeability metrics between all task vectors.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        Dictionary containing pairwise metrics matrix for each metric
    """
    seed_index_everything(cfg)

    dataset_names = list(cfg.mergeability.datasets)
    n_datasets = len(dataset_names)

    if n_datasets < 2:
        raise ValueError(
            f"Need at least 2 datasets for pairwise comparison, got {n_datasets}"
        )
    pylogger.info(f"Computing pairwise mergeability metrics for {n_datasets} datasets")
    pylogger.info(f"Datasets: {dataset_names}")

    # Load pretrained encoder
    pylogger.info("Loading pretrained encoder...")
    pretrained_encoder = load_model_from_hf(model_name=cfg.nn.encoder.model_name)
    pretrained_state_dict = pretrained_encoder.state_dict()

    # Load all fine-tuned models and compute task vectors
    pylogger.info("Loading fine-tuned models and computing task vectors...")
    task_dicts = {}
    for dataset_name in dataset_names:
        pylogger.info(f"  Loading {dataset_name}...")
        finetuned = load_model_from_hf(
            model_name=cfg.nn.encoder.model_name, dataset_name=dataset_name
        )
        task_dicts[dataset_name] = compute_task_dict(
            pretrained_state_dict, finetuned.state_dict()
        )
        del finetuned
        torch.cuda.empty_cache()

    del pretrained_encoder, pretrained_state_dict
    torch.cuda.empty_cache()

    print_memory("after loading models and computing task vectors")

    # Determine which metrics to compute
    metrics_to_compute = list(cfg.mergeability.metrics)
    if metrics_to_compute == ["all"]:
        metrics_to_compute = [k for k in METRIC_REGISTRY.keys() if k != "per_layer_cosine_similarity"]
        if cfg.mergeability.save_per_layer:
            metrics_to_compute.append("per_layer_cosine_similarity")

    pylogger.info(f"Computing metrics: {metrics_to_compute}")

    # Initialize results structure
    # For each metric, we store an NxN matrix (upper triangular)
    results = {
        "model_name": cfg.nn.encoder.model_name,
        "datasets": dataset_names,
        "n_datasets": n_datasets,
        "metrics": {},
    }

    # Initialize matrices for each metric
    for metric_name in metrics_to_compute:
        if metric_name == "per_layer_cosine_similarity":
            # Per-layer metrics stored differently
            results["metrics"][metric_name] = {}
        else:
            # Initialize NxN matrix with None (will be upper triangular)
            results["metrics"][metric_name] = {
                "matrix": [[None for _ in range(n_datasets)] for _ in range(n_datasets)],
                "pairs": {},  # Also store as dict for easy lookup
            }

    # Compute pairwise metrics
    n_pairs = n_datasets * (n_datasets - 1) // 2
    pair_idx = 0

    for i in range(n_datasets):
        for j in range(i + 1, n_datasets):
            pair_idx += 1
            name_i = dataset_names[i]
            name_j = dataset_names[j]
            pair_key = f"{name_i}__{name_j}"

            pylogger.info(f"[{pair_idx}/{n_pairs}] Computing metrics for {name_i} vs {name_j}")

            for metric_name in metrics_to_compute:
                try:
                    metric_value = compute_metric(
                        metric_name,
                        task_dicts[name_i],
                        task_dicts[name_j],
                    )

                    if metric_name == "per_layer_cosine_similarity":
                        results["metrics"][metric_name][pair_key] = metric_value
                    else:
                        # Store in matrix (upper triangular)
                        results["metrics"][metric_name]["matrix"][i][j] = metric_value
                        # Also store in pairs dict for easy lookup
                        results["metrics"][metric_name]["pairs"][pair_key] = metric_value

                except Exception as e:
                    pylogger.error(f"Failed to compute {metric_name} for {pair_key}: {e}")
                    if metric_name != "per_layer_cosine_similarity":
                        results["metrics"][metric_name]["matrix"][i][j] = None
                        results["metrics"][metric_name]["pairs"][pair_key] = None

    # Save results
    output_path = Path(cfg.mergeability.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"pairwise_metrics_{n_datasets}tasks.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    pylogger.info(f"Results saved to {output_file}")

    # Print summary
    pylogger.info("=" * 60)
    pylogger.info("SUMMARY")
    pylogger.info("=" * 60)
    pylogger.info(f"Datasets ({n_datasets}): {dataset_names}")
    pylogger.info(f"Total pairs computed: {n_pairs}")
    pylogger.info("")

    for metric_name in metrics_to_compute:
        if metric_name == "per_layer_cosine_similarity":
            pylogger.info(f"{metric_name}: <per-layer data for {n_pairs} pairs>")
        else:
            pylogger.info(f"{metric_name}:")
            # Print matrix in a readable format
            matrix = results["metrics"][metric_name]["matrix"]
            # Header
            header = "          " + "  ".join(f"{name[:8]:>8}" for name in dataset_names)
            pylogger.info(header)
            for i, name in enumerate(dataset_names):
                row_values = []
                for j in range(n_datasets):
                    if j <= i:
                        row_values.append("    -   ")
                    elif matrix[i][j] is not None:
                        row_values.append(f"{matrix[i][j]:8.4f}")
                    else:
                        row_values.append("   None ")
                pylogger.info(f"{name[:8]:>8}  " + "  ".join(row_values))
            pylogger.info("")

    return results


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="multitask.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
