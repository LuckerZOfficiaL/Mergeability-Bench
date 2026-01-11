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
from model_merging.alignment.rotation_alignment import apply_rotation_alignment
from model_merging.metrics import (
    METRIC_REGISTRY,
    compute_metric,
    compute_all_metrics,
    compute_metric_per_layer,
)
from model_merging.metrics.mergeability import build_calibration_loader
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

    # Load all fine-tuned models
    pylogger.info("Loading fine-tuned models...")
    finetuned_state_dicts = {}
    for dataset_name in dataset_names:
        pylogger.info(f"  Loading {dataset_name}...")
        finetuned = load_model_from_hf(
            model_name=cfg.nn.encoder.model_name, dataset_name=dataset_name
        )
        finetuned_state_dicts[dataset_name] = finetuned.state_dict()
        del finetuned
        torch.cuda.empty_cache()

    # Apply rotation symmetry alignment if enabled
    if cfg.mergeability.get("rot_sym_align", False):
        pylogger.info("Applying rotation symmetry alignment...")
        finetuned_state_dicts = apply_rotation_alignment(
            finetuned_state_dicts=finetuned_state_dicts,
            model_name=cfg.nn.encoder.model_name,
            device=cfg.device,
            logger=pylogger
        )
        pylogger.info("Rotation alignment completed.")

    # Compute task vectors from (potentially aligned) fine-tuned models
    pylogger.info("Computing task vectors...")
    task_dicts = {}
    for dataset_name in dataset_names:
        task_dicts[dataset_name] = compute_task_dict(
            pretrained_state_dict, finetuned_state_dicts[dataset_name]
        )

    # Clean up finetuned state dicts to free memory
    del finetuned_state_dicts
    torch.cuda.empty_cache()

    print_memory("after loading models and computing task vectors")

    # Determine which metrics to compute
    metrics_to_compute = list(cfg.mergeability.metrics)
    if metrics_to_compute == ["all"]:
        metrics_to_compute = list(METRIC_REGISTRY.keys())

    # Check if any activation-based metrics are requested
    activation_metrics = [
        "activation_l2_distance",
        "activation_cosine_similarity",
        "activation_magnitude_ratio",
        "activation_dot_product",
    ]
    needs_activation_data = any(m in metrics_to_compute for m in activation_metrics)

    # Build calibration loader if needed
    calibration_loader = None
    layer_name = None
    if needs_activation_data:
        pylogger.info("Building calibration data loader for activation metrics...")

        # Get calibration settings from config
        n_calibration_samples = cfg.mergeability.get("n_calibration_samples", 10)
        calibration_batch_size = cfg.mergeability.get("calibration_batch_size", 32)
        calibration_random_seed = cfg.mergeability.get("calibration_random_seed", 42)
        layer_name = cfg.mergeability.get("activation_layer_name", "visual.transformer.resblocks.11")

        pylogger.info(f"  Using {n_calibration_samples} samples per dataset (random seed: {calibration_random_seed})")
        pylogger.info(f"  Extracting activations from layer: {layer_name}")

        # Build dataset configs from the benchmark
        # We need to load dataset configs dynamically using OmegaConf
        from omegaconf import OmegaConf

        dataset_configs = []
        config_dir = PROJECT_ROOT / "conf"

        for dataset_name in dataset_names:
            # Load the dataset config file
            dataset_config_path = config_dir / "dataset" / f"{dataset_name}.yaml"
            if dataset_config_path.exists():
                dataset_cfg = OmegaConf.load(dataset_config_path)
                # Note: preprocess_fn will be passed separately to build_calibration_loader
                dataset_configs.append(dataset_cfg)
            else:
                pylogger.warning(f"Dataset config not found: {dataset_config_path}")

        calibration_loader = build_calibration_loader(
            dataset_configs=dataset_configs,
            pretrained_encoder=pretrained_encoder,
            n_samples=n_calibration_samples,
            batch_size=calibration_batch_size,
            device=cfg.device,
            random_seed=calibration_random_seed,
        )

        pylogger.info(f"  Calibration loader built with {len(calibration_loader.dataset)} total samples")

    # Clean up pretrained state dict but keep pretrained_encoder if needed for activation metrics
    if not needs_activation_data:
        del pretrained_encoder, pretrained_state_dict
        torch.cuda.empty_cache()
    else:
        del pretrained_state_dict
        torch.cuda.empty_cache()

    layer_wise = cfg.mergeability.get("layer_wise", False)
    pylogger.info(f"Computing metrics: {metrics_to_compute}")
    pylogger.info(f"Layer-wise mode: {layer_wise}")

    # Initialize results structure
    # For each metric, we store an NxN matrix (upper triangular)
    results = {
        "model_name": cfg.nn.encoder.model_name,
        "datasets": dataset_names,
        "n_datasets": n_datasets,
        "layer_wise": layer_wise,
        "metrics": {},
    }

    # Initialize matrices for each metric
    for metric_name in metrics_to_compute:
        # Initialize NxN matrix with None (will be upper triangular)
        results["metrics"][metric_name] = {
            "matrix": [[None for _ in range(n_datasets)] for _ in range(n_datasets)],
            "pairs": {},  # Also store as dict for easy lookup
        }
        if layer_wise:
            results["metrics"][metric_name]["per_layer"] = {}

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
                    pylogger.info(f"    Computing {metric_name}...")
                    metric_fn = METRIC_REGISTRY[metric_name]

                    # Prepare kwargs for activation metrics
                    kwargs = {}
                    if metric_name in activation_metrics:
                        kwargs = {
                            "pretrained_model": pretrained_encoder,
                            "calibration_loader": calibration_loader,
                            "layer_name": layer_name,
                            "device": cfg.device,
                        }

                    if layer_wise:
                        # Note: Layer-wise computation for activation metrics is not supported
                        # because it would require reconstructing models for each layer
                        if metric_name in activation_metrics:
                            pylogger.warning(
                                f"Layer-wise mode not supported for activation metric {metric_name}, "
                                "computing aggregate value instead"
                            )
                            metric_value = metric_fn(
                                task_dicts[name_i],
                                task_dicts[name_j],
                                **kwargs
                            )
                            results["metrics"][metric_name]["matrix"][i][j] = metric_value
                            results["metrics"][metric_name]["pairs"][pair_key] = metric_value
                            pylogger.info(f"      ✓ {metric_name} = {metric_value}")
                        else:
                            # Compute per-layer and average for weight-based metrics
                            per_layer_values = compute_metric_per_layer(
                                metric_fn,
                                task_dicts[name_i],
                                task_dicts[name_j],
                            )
                            import math
                            valid_values = [v for v in per_layer_values.values() if not math.isnan(v)]
                            avg_value = sum(valid_values) / len(valid_values) if valid_values else 0.0

                            # Store average in matrix
                            results["metrics"][metric_name]["matrix"][i][j] = avg_value
                            results["metrics"][metric_name]["pairs"][pair_key] = avg_value
                            # Store per-layer breakdown
                            results["metrics"][metric_name]["per_layer"][pair_key] = per_layer_values
                            pylogger.info(f"      ✓ {metric_name} = {avg_value} (avg of {len(valid_values)} layers)")
                    else:
                        # Normal aggregate computation
                        metric_value = metric_fn(
                            task_dicts[name_i],
                            task_dicts[name_j],
                            **kwargs
                        )
                        results["metrics"][metric_name]["matrix"][i][j] = metric_value
                        results["metrics"][metric_name]["pairs"][pair_key] = metric_value
                        pylogger.info(f"      ✓ {metric_name} = {metric_value}")

                except Exception as e:
                    pylogger.error(f"Failed to compute {metric_name} for {pair_key}: {e}")
                    import traceback
                    pylogger.error(traceback.format_exc())
                    results["metrics"][metric_name]["matrix"][i][j] = None
                    results["metrics"][metric_name]["pairs"][pair_key] = None

    # Clean up activation resources if they were used
    if needs_activation_data:
        del pretrained_encoder, calibration_loader
        torch.cuda.empty_cache()

    # Save results
    output_path = Path(cfg.mergeability.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use benchmark_name if provided, otherwise fall back to listing datasets
    benchmark_name = cfg.mergeability.get("benchmark_name", None)
    rot_sym_align = cfg.mergeability.get("rot_sym_align", False)

    # Add rotation alignment suffix if enabled
    rot_suffix = "_rot_aligned" if rot_sym_align else ""

    if benchmark_name:
        output_file = output_path / f"pairwise_metrics_{benchmark_name}{rot_suffix}.json"
    else:
        datasets_suffix = "_".join(dataset_names)
        output_file = output_path / f"pairwise_metrics_{n_datasets}tasks_{datasets_suffix}{rot_suffix}.json"

    # Load existing results if file exists and merge new metrics
    if output_file.exists():
        pylogger.info(f"Loading existing results from {output_file}")
        with open(output_file, "r") as f:
            existing_results = json.load(f)

        # Verify compatibility
        if existing_results.get("model_name") != results["model_name"]:
            pylogger.warning(
                f"Model name mismatch: existing={existing_results.get('model_name')}, "
                f"current={results['model_name']}"
            )
        if existing_results.get("datasets") != results["datasets"]:
            pylogger.warning(
                f"Dataset list mismatch: existing={existing_results.get('datasets')}, "
                f"current={results['datasets']}"
            )

        # Merge metrics: new metrics will overwrite existing ones if they have the same name
        for metric_name, metric_data in results["metrics"].items():
            if metric_name in existing_results["metrics"]:
                pylogger.info(f"Overwriting existing metric: {metric_name}")
            else:
                pylogger.info(f"Adding new metric: {metric_name}")
            existing_results["metrics"][metric_name] = metric_data

        # Use the merged results
        results = existing_results
        pylogger.info(f"Merged results now contain {len(results['metrics'])} metrics")
    else:
        pylogger.info(f"No existing results found, creating new file")

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
        suffix = " (layer-wise avg)" if layer_wise else ""
        pylogger.info(f"{metric_name}{suffix}:")
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
