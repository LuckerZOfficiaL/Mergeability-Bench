"""
Quick test script for gradient-based mergeability metrics.

This script tests the gradient metric computation on a small pair of tasks.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from omegaconf import OmegaConf

from model_merging.metrics.mergeability import (
    encoder_gradient_cosine_similarity,
    encoder_gradient_l2_distance,
    encoder_gradient_dot_product,
    input_gradient_cosine_similarity,
    input_gradient_l2_distance,
    input_gradient_dot_product,
)
from model_merging.utils.io_utils import load_model_from_hf
from model_merging.utils.utils import compute_task_dict

# Configuration
MODEL_NAME = "openai/clip-vit-base-patch32"
DATASET_1 = "MNIST"
DATASET_2 = "CIFAR10"
N_SAMPLES = 5  # Small number for quick testing
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Testing gradient metrics on {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Datasets: {DATASET_1} vs {DATASET_2}")
print(f"Calibration samples: {N_SAMPLES} per dataset")
print("=" * 60)

# Load pretrained encoder
print("\n1. Loading pretrained encoder...")
pretrained_encoder = load_model_from_hf(model_name=MODEL_NAME)
pretrained_state_dict = pretrained_encoder.state_dict()
print(f"   ✓ Pretrained encoder loaded")

# Load fine-tuned models
print(f"\n2. Loading fine-tuned models...")
print(f"   Loading {DATASET_1}...")
finetuned_1 = load_model_from_hf(model_name=MODEL_NAME, dataset_name=DATASET_1)
finetuned_state_dict_1 = finetuned_1.state_dict()
del finetuned_1
torch.cuda.empty_cache()

print(f"   Loading {DATASET_2}...")
finetuned_2 = load_model_from_hf(model_name=MODEL_NAME, dataset_name=DATASET_2)
finetuned_state_dict_2 = finetuned_2.state_dict()
del finetuned_2
torch.cuda.empty_cache()
print(f"   ✓ Fine-tuned models loaded")

# Compute task vectors
print("\n3. Computing task vectors...")
task_dict_1 = compute_task_dict(pretrained_state_dict, finetuned_state_dict_1)
task_dict_2 = compute_task_dict(pretrained_state_dict, finetuned_state_dict_2)
print(f"   ✓ Task vectors computed")

# Clean up
del finetuned_state_dict_1, finetuned_state_dict_2
torch.cuda.empty_cache()

# Load dataset configs
print("\n4. Loading dataset configs...")
config_dir = Path(__file__).parent.parent / "conf"
dataset_config_1_path = config_dir / "dataset" / f"{DATASET_1}.yaml"
dataset_config_2_path = config_dir / "dataset" / f"{DATASET_2}.yaml"

dataset_config_1 = OmegaConf.load(dataset_config_1_path)
dataset_config_2 = OmegaConf.load(dataset_config_2_path)
print(f"   ✓ Dataset configs loaded")

# Test encoder gradient metrics
print("\n5. Testing ENCODER gradient metrics...")
print("   (This may take a minute...)")

try:
    print("\n   5a. Encoder gradient cosine similarity...")
    sim = encoder_gradient_cosine_similarity(
        task_dict_1,
        task_dict_2,
        pretrained_model=pretrained_encoder,
        dataset_config_1=dataset_config_1,
        dataset_config_2=dataset_config_2,
        n_calibration_samples=N_SAMPLES,
        calibration_batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    print(f"       ✓ Cosine similarity: {sim:.4f}")
except Exception as e:
    print(f"       ✗ Error: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n   5b. Encoder gradient L2 distance...")
    dist = encoder_gradient_l2_distance(
        task_dict_1,
        task_dict_2,
        pretrained_model=pretrained_encoder,
        dataset_config_1=dataset_config_1,
        dataset_config_2=dataset_config_2,
        n_calibration_samples=N_SAMPLES,
        calibration_batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    print(f"       ✓ L2 distance: {dist:.4f}")
except Exception as e:
    print(f"       ✗ Error: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n   5c. Encoder gradient dot product...")
    dot = encoder_gradient_dot_product(
        task_dict_1,
        task_dict_2,
        pretrained_model=pretrained_encoder,
        dataset_config_1=dataset_config_1,
        dataset_config_2=dataset_config_2,
        n_calibration_samples=N_SAMPLES,
        calibration_batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    print(f"       ✓ Dot product: {dot:.4f}")
except Exception as e:
    print(f"       ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test input gradient metrics
print("\n6. Testing INPUT gradient metrics...")
print("   (This may take a minute...)")

try:
    print("\n   6a. Input gradient cosine similarity...")
    sim = input_gradient_cosine_similarity(
        task_dict_1,
        task_dict_2,
        pretrained_model=pretrained_encoder,
        dataset_config_1=dataset_config_1,
        dataset_config_2=dataset_config_2,
        n_calibration_samples=N_SAMPLES,
        calibration_batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    print(f"       ✓ Cosine similarity: {sim:.4f}")
except Exception as e:
    print(f"       ✗ Error: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n   6b. Input gradient L2 distance...")
    dist = input_gradient_l2_distance(
        task_dict_1,
        task_dict_2,
        pretrained_model=pretrained_encoder,
        dataset_config_1=dataset_config_1,
        dataset_config_2=dataset_config_2,
        n_calibration_samples=N_SAMPLES,
        calibration_batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    print(f"       ✓ L2 distance: {dist:.4f}")
except Exception as e:
    print(f"       ✗ Error: {e}")
    import traceback
    traceback.print_exc()

try:
    print("\n   6c. Input gradient dot product...")
    dot = input_gradient_dot_product(
        task_dict_1,
        task_dict_2,
        pretrained_model=pretrained_encoder,
        dataset_config_1=dataset_config_1,
        dataset_config_2=dataset_config_2,
        n_calibration_samples=N_SAMPLES,
        calibration_batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    print(f"       ✓ Dot product: {dot:.4f}")
except Exception as e:
    print(f"       ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Testing complete!")
