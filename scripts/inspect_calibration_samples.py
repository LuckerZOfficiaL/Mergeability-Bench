"""Inspect which samples are used in the calibration set."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omegaconf import OmegaConf
from hydra.utils import instantiate
from model_merging.data.dataset import load_dataset
import open_clip
import torch


def main():
    # Load pretrained model to get preprocessor
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    preprocess_fn = preprocess

    # N8 benchmark datasets
    dataset_names = ["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"]
    n_samples = 10
    random_seed = 42

    import random
    random.seed(random_seed)

    print("=" * 80)
    print("CALIBRATION SAMPLES INSPECTION")
    print("=" * 80)
    print(f"\nNumber of samples per dataset: {n_samples}")
    print(f"Total datasets: {len(dataset_names)}")
    print(f"Total calibration samples: {n_samples * len(dataset_names)}\n")

    all_sample_info = []

    for dataset_name in dataset_names:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*80}")

        # Load dataset config
        dataset_config_path = PROJECT_ROOT / "conf" / "dataset" / f"{dataset_name}.yaml"
        if not dataset_config_path.exists():
            print(f"  Config not found: {dataset_config_path}")
            continue

        dataset_cfg = OmegaConf.load(dataset_config_path)

        try:
            # Instantiate the HF dataset
            hf_dataset = instantiate(dataset_cfg.hf_dataset)

            # Load the dataset
            dataset = load_dataset(
                name=dataset_cfg.name,
                hf_dataset=hf_dataset,
                preprocess_fn=preprocess_fn,
                ft_epochs=dataset_cfg.get("ft_epochs", 10),
                split_map=dataset_cfg.get("split_map", None),
                batch_size=32,
                label_map=dataset_cfg.get("label_map", None),
                classnames_override=dataset_cfg.get("classnames_override", None),
            )

            test_dataset = dataset.test_dataset
            n_available = len(test_dataset)
            n_to_sample = min(n_samples, n_available)

            # Random sampling with fixed seed for reproducibility
            indices = random.sample(range(n_available), n_to_sample)
            indices.sort()  # Sort for consistent ordering

            print(f"  Total validation samples available: {n_available}")
            print(f"  Samples taken: {n_to_sample}")
            print(f"\n  Sample indices and labels:")

            for idx in indices:
                sample = test_dataset[idx]
                # sample is typically (image, label)
                if isinstance(sample, tuple) and len(sample) >= 2:
                    _, label = sample[0], sample[1]
                    if hasattr(dataset, 'classnames') and dataset.classnames:
                        class_name = dataset.classnames[label] if label < len(dataset.classnames) else f"class_{label}"
                        print(f"    Index {idx:2d}: Label {label:3d} ({class_name})")
                    else:
                        print(f"    Index {idx:2d}: Label {label:3d}")

                    all_sample_info.append({
                        'dataset': dataset_name,
                        'index': idx,
                        'label': label
                    })
                else:
                    print(f"    Index {idx:2d}: (unable to extract label)")

        except Exception as e:
            print(f"  Error loading dataset: {e}")
            continue

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples collected: {len(all_sample_info)}")

    # Show label distribution per dataset
    print("\nLabel distribution per dataset:")
    for dataset_name in dataset_names:
        dataset_samples = [s for s in all_sample_info if s['dataset'] == dataset_name]
        if dataset_samples:
            labels = [s['label'] for s in dataset_samples]
            unique_labels = len(set(labels))
            print(f"  {dataset_name:12s}: {len(dataset_samples)} samples, {unique_labels} unique classes")
            print(f"    Labels: {labels}")


if __name__ == "__main__":
    main()
