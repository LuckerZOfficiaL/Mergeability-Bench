#!/usr/bin/env python3
"""Download model artifacts from W&B"""
import wandb
import os
from pathlib import Path

# Configuration
ENTITY = "lzhou00-sapienza-universit-di-roma"
PROJECT = "model_merging"
CHECKPOINT_DIR = "/home/ubuntu/thesis/MM/Mergeability-Bench/checkpoints/ViT-B-16"

# Models to download (all 20 datasets with _moderate_update_grad_magnitude)
MODELS = [
    "MNIST_moderate_update_grad_magnitude",
    "Cars_moderate_update_grad_magnitude",
    "CIFAR10_moderate_update_grad_magnitude",
    "CIFAR100_moderate_update_grad_magnitude",
    "DTD_moderate_update_grad_magnitude",
    "EMNIST_moderate_update_grad_magnitude",
    "EuroSAT_moderate_update_grad_magnitude",
    "FER2013_moderate_update_grad_magnitude",
    "FashionMNIST_moderate_update_grad_magnitude",
    "Flowers102_moderate_update_grad_magnitude",
    "Food101_moderate_update_grad_magnitude",
    "GTSRB_moderate_update_grad_magnitude",
    "KMNIST_moderate_update_grad_magnitude",
    "OxfordIIITPet_moderate_update_grad_magnitude",
    "PCAM_moderate_update_grad_magnitude",
    "RESISC45_moderate_update_grad_magnitude",
    "RenderedSST2_moderate_update_grad_magnitude",
    "STL10_moderate_update_grad_magnitude",
    "SUN397_moderate_update_grad_magnitude",
    "SVHN_moderate_update_grad_magnitude"
]

def download_artifact(artifact_name, download_dir):
    """Download a single artifact from W&B"""
    api = wandb.Api()

    # Try different artifact naming patterns
    patterns = [
        f"{ENTITY}/{PROJECT}/{artifact_name}:latest",
        f"{ENTITY}/{PROJECT}/ViT-B-16-{artifact_name}:latest",
        f"{ENTITY}/{PROJECT}/model-{artifact_name}:latest",
    ]

    for pattern in patterns:
        try:
            print(f"Trying to download: {pattern}")
            artifact = api.artifact(pattern)
            artifact_dir = artifact.download(root=download_dir)
            print(f"✓ Downloaded {artifact_name} to {artifact_dir}")
            return True
        except Exception as e:
            print(f"  Failed with pattern {pattern}: {e}")
            continue

    print(f"✗ Could not download {artifact_name} with any pattern")
    return False

def main():
    print(f"Downloading artifacts to {CHECKPOINT_DIR}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for model_name in MODELS:
        download_dir = os.path.join(CHECKPOINT_DIR, model_name)
        os.makedirs(download_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Downloading {model_name}")
        print(f"{'='*60}")
        download_artifact(model_name, download_dir)

if __name__ == "__main__":
    main()