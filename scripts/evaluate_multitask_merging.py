import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from model_merging.data.dataset import HFImageClassification
from model_merging.model.image_classifier import ImageClassifier
import open_clip
import wandb

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import model_merging  # noqa
from model_merging.model.encoder import ClassificationHead, ImageEncoder
from model_merging.model.heads import (
    get_classification_head,
)
from model_merging.utils.io_utils import (
    boilerplate,
    load_model_from_hf,
)
from model_merging.utils.plots import plot_interactive_radar_chart
from model_merging.utils.utils import (
    build_callbacks,
    get_finetuning_accuracies,
    compute_avg_accuracy,
    print_memory,
)
import json
import os

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def load_config(
    config_path: str,
    config_name: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    """
    Load a Hydra config without launching a full Hydra app.

    Args:
        config_path (str): Path to the folder containing your configs (relative to project root).
        config_name (str): Name of the YAML config file (without `.yaml`).
        overrides (list[str], optional): List of override strings, e.g. ["trainer.max_epochs=20"].

    Returns:
        DictConfig: The loaded configuration.
    """
    overrides = overrides or []
    abs_config_path = str(Path(config_path).absolute())

    with hydra.initialize(config_path=abs_config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)

    return cfg


def run_single(cfg: DictConfig, datasets_to_use: Optional[List] = None, pair_name: Optional[str] = None) -> Dict:
    """Run merging evaluation for a single set of datasets.

    Args:
        cfg: run configuration, defined by Hydra in /conf
        datasets_to_use: Optional list of dataset configs to use (for pairwise mode)
        pair_name: Optional name for the pair (for logging/saving)

    Returns:
        Dictionary containing evaluation results
    """
    seed_index_everything(cfg)

    logger, template_core = boilerplate(cfg)

    # Use provided datasets or fall back to config
    datasets = datasets_to_use if datasets_to_use is not None else list(cfg.benchmark.datasets)
    num_tasks = len(datasets)

    # Temporarily disable struct mode to allow dynamic update
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.num_tasks = num_tasks  # Now we can safely update it
    omegaconf.OmegaConf.set_struct(cfg, True)  # Re-enable struct mode

    # upperbound accuracies, used for logging the normalized accuracy
    finetuned_accuracies: Dict[str, float] = get_finetuning_accuracies(
        cfg.misc.finetuned_accuracy_path
    )[cfg.nn.encoder.model_name]

    # only has vision encoder, no text transformer
    zeroshot_encoder: ImageEncoder = load_model_from_hf(
        model_name=cfg.nn.encoder.model_name
    )

    finetuned_models = {
        dataset: load_model_from_hf(
            model_name=cfg.nn.encoder.model_name, dataset_name=dataset.name
        ).state_dict()
        for dataset in datasets
    }

    if pair_name:
        pylogger.info(f"=== Evaluating pair: {pair_name} ===")
    pylogger.info(f"Number of tasks: {num_tasks}")
    pylogger.info(f"Finetuned models: {[d.name for d in datasets]}")
    pylogger.info(f"Using merger: {cfg.merger._target_}")

    merger = instantiate(cfg.merger)

    merged_encoder = merger.merge(zeroshot_encoder, finetuned_models)

    logger.log_configuration(merged_encoder, cfg)

    results = {}
    print_memory("before eval")
    for dataset_cfg in datasets:

        dataset = instantiate(
            dataset_cfg, preprocess_fn=zeroshot_encoder.val_preprocess
        )

        classification_head = get_classification_head(
            cfg.nn.encoder.model_name,
            dataset_cfg.name,
            ckpt_path=cfg.misc.ckpt_path,
            openclip_cachedir=cfg.misc.openclip_cachedir,
            device=cfg.device,
        )

        model = ImageClassifier(
            encoder=merged_encoder,
            classifier=classification_head,
            x_key=cfg.conventions.x_key,
            y_key=cfg.conventions.y_key,
        )

        model.set_metrics(len(dataset.classnames))
        model.set_task(dataset_cfg.name)
        model.set_finetuning_accuracy(
            finetuned_accuracies[
                dataset_cfg.name + "Val" if cfg.eval_on_train else dataset_cfg.name
            ]
        )

        callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

        trainer = pl.Trainer(
            default_root_dir=cfg.core.storage_dir,
            plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
            logger=logger,
            callbacks=callbacks,
            limit_test_batches=(
                cfg.number_of_train_batches if cfg.eval_on_train else None
            ),
            **cfg.train.trainer,
        )

        if cfg.eval_on_train:
            pylogger.error("For now evaluation supported only on val-set")
            pylogger.info(f"Evaluating on {dataset_cfg.name} the training set")
            test_results = trainer.test(model=model, dataloaders=dataset.train_loader)

        else:
            pylogger.info(f"Evaluating on the {dataset_cfg.name} test set!")
            test_results = trainer.test(model=model, dataloaders=dataset.test_loader)

        results[dataset_cfg.name] = test_results

    avg = compute_avg_accuracy(results)
    results["avg"] = [
        avg
    ]  # as a list for consistency due to lightning logging stuff this way

    logger.experiment.log(avg)

    pylogger.info(results)

    # Extract merger name from target (e.g., "model_merging.merger.weight_avg_merger.WeightAvgMerger" -> "weight_avg")
    merger_name = cfg.merger._target_.split(".")[-2].replace("_merger", "")

    # Create merger-specific folder
    results_path = Path(cfg.misc.results_path) / merger_name
    results_path.mkdir(parents=True, exist_ok=True)

    # Use pair_name for filename if provided, otherwise use num_tasks
    if pair_name:
        filename = f"pair_{pair_name}.json"
    else:
        filename = f"{num_tasks}.json"

    with open(results_path / filename, "w+") as f:
        json.dump(results, f, indent=4)

    radarchart = plot_interactive_radar_chart(results, title="Radar Chart")
    logger.experiment.log({"radar": wandb.Plotly(radarchart)})

    pylogger.info(f"Results saved to {results_path / filename}")

    logger.experiment.log_artifact(
        wandb.Artifact(
            f"results_{cfg.nn.encoder.model_name}_{pair_name or num_tasks}",
            type="results",
            metadata={"results": str(results_path)},
        )
    )

    if logger is not None:
        logger.experiment.finish()

    return results


def run(cfg: DictConfig):
    """Main entry point - handles both single run and all_pairwise mode.

    Args:
        cfg: run configuration, defined by Hydra in /conf
    """
    all_pairwise = cfg.get("all_pairwise", False)

    if not all_pairwise:
        # Standard single run with all datasets in benchmark
        return run_single(cfg)

    # All pairwise mode: run merging for each pair of datasets
    datasets = list(cfg.benchmark.datasets)
    n_datasets = len(datasets)
    n_pairs = n_datasets * (n_datasets - 1) // 2

    pylogger.info(f"Running ALL PAIRWISE merging evaluation")
    pylogger.info(f"Benchmark has {n_datasets} datasets: {[d.name for d in datasets]}")
    pylogger.info(f"Total pairs to evaluate: {n_pairs}")

    all_results = {}
    pair_idx = 0

    for i in range(n_datasets):
        for j in range(i + 1, n_datasets):
            pair_idx += 1
            dataset_i = datasets[i]
            dataset_j = datasets[j]
            pair_name = f"{dataset_i.name}__{dataset_j.name}"

            pylogger.info(f"\n{'='*60}")
            pylogger.info(f"[{pair_idx}/{n_pairs}] Evaluating pair: {pair_name}")
            pylogger.info(f"{'='*60}\n")

            try:
                pair_results = run_single(
                    cfg,
                    datasets_to_use=[dataset_i, dataset_j],
                    pair_name=pair_name
                )
                all_results[pair_name] = pair_results
            except Exception as e:
                pylogger.error(f"Failed to evaluate pair {pair_name}: {e}")
                all_results[pair_name] = {"error": str(e)}

    # Save summary of all pairwise results
    # Extract merger name from target (e.g., "model_merging.merger.weight_avg_merger.WeightAvgMerger" -> "weight_avg")
    merger_name = cfg.merger._target_.split(".")[-2].replace("_merger", "")

    # Create merger-specific folder
    results_path = Path(cfg.misc.results_path) / merger_name
    results_path.mkdir(parents=True, exist_ok=True)

    # Get benchmark name from config (e.g., "N8", "N20")
    benchmark_name = cfg.benchmark.get("name", f"N{n_datasets}")
    summary_file = results_path / f"all_pairwise_summary_{benchmark_name}.json"
    with open(summary_file, "w+") as f:
        json.dump(all_results, f, indent=4)

    pylogger.info(f"\n{'='*60}")
    pylogger.info(f"ALL PAIRWISE EVALUATION COMPLETE")
    pylogger.info(f"Summary saved to: {summary_file}")
    pylogger.info(f"{'='*60}")

    return all_results


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="multitask.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
