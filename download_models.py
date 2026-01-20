#!/usr/bin/env python3
"""Download models from wandb to local checkpoints directory."""

import os
import wandb
from pathlib import Path

# Initialize wandb API
api = wandb.Api()

# Base checkpoint directory
base_dir = Path("/home/lzhou/MM/Mergeability-Bench/checkpoints/ViT-B-16")
base_dir.mkdir(parents=True, exist_ok=True)

# Get all runs from your project
runs = api.runs('lzhou00-sapienza-universit-di-roma/model_merging')

# Filter for upload runs
upload_runs = [r for r in runs if 'upload' in r.name.lower()]

print(f"Found {len(upload_runs)} upload runs")

# Download models for each regularization setup
for run in upload_runs:
    print(f"\nProcessing: {run.name}")

    # Extract dataset name and suffix from run name
    # Format: upload_<dataset>_<suffix>
    parts = run.name.replace('upload_', '').split('_', 1)
    if len(parts) < 2:
        print(f"  Skipping {run.name} - unexpected format")
        continue

    dataset_name = parts[0]
    suffix = '_' + parts[1]

    # Create target directory: checkpoints/ViT-B-16/<dataset>_<suffix>/
    target_dir = base_dir / f"{dataset_name}{suffix}"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Try to download artifacts
    try:
        artifacts = list(run.logged_artifacts())
        if artifacts:
            for artifact in artifacts:
                print(f"  Found artifact: {artifact.name} (type: {artifact.type})")

                # Download the artifact
                artifact_dir = artifact.download(root=str(target_dir / "artifact_temp"))

                # Look for model files and move them
                for root, dirs, files in os.walk(artifact_dir):
                    for file in files:
                        if file.endswith(('.pt', '.pth', '.bin', '.safetensors')):
                            src = os.path.join(root, file)
                            dst = target_dir / "model.pt"
                            print(f"    Moving {file} to {dst}")
                            os.rename(src, dst)
                            break

                # Clean up temp directory
                import shutil
                if os.path.exists(target_dir / "artifact_temp"):
                    shutil.rmtree(target_dir / "artifact_temp")

                print(f"  ✓ Downloaded to {target_dir}")
        else:
            print(f"  No artifacts found")
    except Exception as e:
        print(f"  Error: {e}")

print("\n✓ Download complete!")
