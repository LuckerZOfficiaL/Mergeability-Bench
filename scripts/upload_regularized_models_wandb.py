#!/usr/bin/env python3
"""
Upload regularized finetuned models to Weights & Biases as artifacts.

Usage:
    python scripts/upload_regularized_models_wandb.py --reg_type both  # For R2+R3
    python scripts/upload_regularized_models_wandb.py --reg_type r2    # For R2 only
    python scripts/upload_regularized_models_wandb.py --reg_type r3    # For R3 only
"""

import os
import argparse
from pathlib import Path
import wandb

# N20 dataset names
N20_DATASETS = [
    "Cars", "CIFAR10", "CIFAR100", "DTD", "EMNIST",
    "EuroSAT", "FashionMNIST", "FER2013", "Flowers102", "Food101",
    "GTSRB", "KMNIST", "MNIST", "OxfordIIITPet", "PCAM",
    "RenderedSST2", "RESISC45", "SUN397", "SVHN", "STL10"
]

def get_suffix_and_dir(reg_type: str):
    """Get the suffix and directory name based on regularization type."""
    if reg_type == "both":
        return "_moderate_update_grad_magnitude", "grad_magnitude_moderate_update"
    elif reg_type == "r2":
        return "_moderate_update", "moderate_update"
    elif reg_type == "r3":
        return "_grad_magnitude", "grad_magnitude"
    else:
        raise ValueError(f"Unknown reg_type: {reg_type}. Use 'both', 'r2', or 'r3'")

def upload_model_to_wandb(
    checkpoint_path: str,
    model_name: str,
    dataset_name: str,
    project: str,
    entity: str = None,
    reg_type: str = "both"
):
    """
    Upload a single model checkpoint to W&B as an artifact.

    Args:
        checkpoint_path: Path to the model.pt file
        model_name: Base model name (e.g., "ViT-B-16")
        dataset_name: Dataset name with suffix (e.g., "MNIST_moderate_update_grad_magnitude")
        project: W&B project name
        entity: W&B entity (username or team name), optional
        reg_type: Type of regularization for metadata
    """
    # Create artifact name
    artifact_name = f"{model_name}-{dataset_name}"

    print(f"\nUploading {dataset_name}...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Artifact: {artifact_name}")

    try:
        # Initialize W&B run
        run = wandb.init(
            project=project,
            entity=entity,
            name=f"upload_{dataset_name}",
            job_type="upload-checkpoint",
            tags=["finetuned", "regularized", reg_type, dataset_name.split('_')[0]]
        )

        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"{model_name} finetuned on {dataset_name.split('_')[0]} with regularization",
            metadata={
                "model_name": model_name,
                "dataset": dataset_name.split('_')[0],
                "regularization_type": reg_type,
                "has_r2_moderate_update": "moderate_update" in dataset_name,
                "has_r3_grad_magnitude": "grad_magnitude" in dataset_name,
                "lambda_moderate_update": 0.01 if "moderate_update" in dataset_name else 0.0,
                "lambda_grad_magnitude": 0.001 if "grad_magnitude" in dataset_name else 0.0,
            }
        )

        # Add the checkpoint file
        artifact.add_file(checkpoint_path, name="model.pt")

        # Log the artifact
        run.log_artifact(artifact)

        print(f"  ✓ Artifact uploaded successfully")
        print(f"  URL: {run.url}")

        # Finish the run
        run.finish()

        return True

    except Exception as e:
        print(f"  ✗ Failed to upload: {e}")
        if 'run' in locals():
            run.finish()
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload regularized models to Weights & Biases")
    parser.add_argument(
        "--reg_type",
        type=str,
        required=True,
        choices=["both", "r2", "r3"],
        help="Type of regularization: 'both' (R2+R3), 'r2' (R2 only), 'r3' (R3 only)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-B-16",
        help="Base model name (default: ViT-B-16)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="model_merging",
        help="W&B project name (default: model_merging)"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity (username or team name), optional"
    )
    parser.add_argument(
        "--checkpoint_base_dir",
        type=str,
        default="/home/ubuntu/thesis/MM/model-merging/checkpoints/ViT-B-16",
        help="Base directory containing checkpoints"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to upload (default: all N20 datasets)"
    )

    args = parser.parse_args()

    # Get suffix and directory for this regularization type
    suffix, dir_name = get_suffix_and_dir(args.reg_type)

    # Determine which datasets to upload
    datasets_to_upload = args.datasets if args.datasets else N20_DATASETS

    print("="*70)
    print(f"Uploading regularized models to Weights & Biases")
    print(f"  Project: {args.project}")
    print(f"  Entity: {args.entity or 'default'}")
    print(f"  Regularization type: {args.reg_type}")
    print(f"  Suffix: {suffix}")
    print(f"  Directory: {dir_name}")
    print(f"  Datasets: {len(datasets_to_upload)} total")
    print("="*70)

    # Upload each dataset
    success_count = 0
    failed_datasets = []

    for i, dataset in enumerate(datasets_to_upload, 1):
        print(f"\n[{i}/{len(datasets_to_upload)}] Processing {dataset}...")

        # Construct checkpoint path
        dataset_name_with_suffix = f"{dataset}{suffix}"
        checkpoint_path = os.path.join(
            args.checkpoint_base_dir,
            dir_name,
            dataset_name_with_suffix,
            "model.pt"
        )

        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"  ✗ Checkpoint not found: {checkpoint_path}")
            failed_datasets.append(dataset)
            continue

        # Upload the model
        success = upload_model_to_wandb(
            checkpoint_path=checkpoint_path,
            model_name=args.model_name,
            dataset_name=dataset_name_with_suffix,
            project=args.project,
            entity=args.entity,
            reg_type=args.reg_type
        )

        if success:
            success_count += 1
        else:
            failed_datasets.append(dataset)

    # Print summary
    print("\n" + "="*70)
    print("UPLOAD SUMMARY")
    print("="*70)
    print(f"Total datasets: {len(datasets_to_upload)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_datasets)}")

    if failed_datasets:
        print(f"\nFailed datasets:")
        for dataset in failed_datasets:
            print(f"  - {dataset}")

    print("="*70)

    if success_count > 0:
        project_url = f"https://wandb.ai/{args.entity or 'YOUR_USERNAME'}/{args.project}/artifacts"
        print(f"\nView artifacts at: {project_url}")

if __name__ == "__main__":
    main()
