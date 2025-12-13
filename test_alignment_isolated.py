"""
Isolated test script for rotation alignment.
Tests alignment on just 2 models to debug any issues.
"""

import sys
sys.path.insert(0, 'src')

import torch
import logging
from model_merging.utils.io_utils import load_model_from_hf
from model_merging.alignment.rotation_alignment import apply_rotation_alignment

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("="*70)
    logger.info("ISOLATED ROTATION ALIGNMENT TEST")
    logger.info("="*70)

    model_name = 'ViT-B-16'

    # Create fake dataset config objects (mimics what evaluate_multitask_merging does)
    class FakeDatasetConfig:
        def __init__(self, name):
            self.name = name

    dataset_mnist = FakeDatasetConfig('MNIST')
    dataset_cars = FakeDatasetConfig('Cars')

    # Load models
    logger.info("\n[1] Loading MNIST model...")
    mnist_model = load_model_from_hf(model_name=model_name, dataset_name='MNIST')
    mnist_state_dict = mnist_model.state_dict()
    logger.info(f"    Loaded: {len(mnist_state_dict)} parameters")

    logger.info("\n[2] Loading Cars model...")
    cars_model = load_model_from_hf(model_name=model_name, dataset_name='Cars')
    cars_state_dict = cars_model.state_dict()
    logger.info(f"    Loaded: {len(cars_state_dict)} parameters")

    # Create finetuned_models dict (same structure as in evaluate_multitask_merging)
    finetuned_models = {
        dataset_mnist: mnist_state_dict,
        dataset_cars: cars_state_dict,
    }

    logger.info("\n[3] Applying rotation alignment...")
    logger.info("    This should align Cars to MNIST (anchor)")

    # Apply alignment
    aligned_models = apply_rotation_alignment(
        finetuned_state_dicts=finetuned_models,
        model_name=model_name,
        device='cpu',
        logger=logger
    )

    logger.info("\n[4] Checking results...")
    logger.info(f"    Number of models returned: {len(aligned_models)}")
    logger.info(f"    Keys: {[k.name for k in aligned_models.keys()]}")

    # Verify the aligned Cars model is different from original
    original_cars = finetuned_models[dataset_cars]
    aligned_cars = aligned_models[dataset_cars]

    num_different = 0
    for (k1, v1), (k2, v2) in zip(original_cars.items(), aligned_cars.items()):
        if not torch.allclose(v1, v2, atol=1e-6):
            num_different += 1

    logger.info(f"    Parameters changed in Cars: {num_different}/{len(original_cars)}")

    if num_different > 0:
        logger.info("\n✓ SUCCESS: Alignment completed and modified parameters!")
    else:
        logger.warning("\n✗ WARNING: No parameters were changed!")

    logger.info("\n" + "="*70)
    logger.info("TEST COMPLETE")
    logger.info("="*70)

if __name__ == '__main__':
    main()
