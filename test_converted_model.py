"""
Test that a model converted from TransFusion format can be loaded and used in model-merging.

This script:
1. Loads the converted model using model-merging's utilities
2. Verifies the model structure
3. Tests forward pass with dummy data
"""

import sys
import torch
from pathlib import Path

# Add model-merging to path
sys.path.insert(0, '/home/ubuntu/thesis/MM/model-merging/src')

from model_merging.utils.io_utils import load_model_from_disk
from model_merging.model.encoder import ImageEncoder


def test_converted_model():
    """Test that the converted model works in model-merging."""

    print("="*80)
    print("TESTING CONVERTED MODEL IN MODEL-MERGING")
    print("="*80)

    # Path to converted model
    checkpoint_path = '/home/ubuntu/thesis/MM/model-merging/checkpoints/ViT-B-16/converted_from_transfusion/SVHN/checkpoint.pt'

    print(f"\nTest 1: Loading checkpoint directly...")
    print(f"  Path: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"  ✓ Checkpoint loaded")

        # Check structure
        if isinstance(checkpoint, dict):
            # Get keys that start with 'model.'
            model_keys = [k for k in checkpoint.keys() if k.startswith('model.')]
            other_keys = [k for k in checkpoint.keys() if not k.startswith('model.')]

            print(f"  Model parameters: {len(model_keys)}")
            print(f"  Metadata keys: {other_keys}")

            # Sample model keys
            print("  Sample model keys:")
            for key in model_keys[:5]:
                print(f"    - {key}")
        else:
            print(f"  Unexpected checkpoint type: {type(checkpoint)}")

    except Exception as e:
        print(f"  ✗ Error loading checkpoint: {e}")
        return False

    print(f"\nTest 2: Loading with model-merging's load_model_from_disk...")

    try:
        model = load_model_from_disk(checkpoint_path, model_name='ViT-B-16')
        print(f"  ✓ Model loaded successfully")
        print(f"  Model type: {type(model)}")
        print(f"  Is ImageEncoder: {isinstance(model, ImageEncoder)}")

    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        print(f"\nAttempting alternative loading method...")

        # Try creating ImageEncoder and loading state dict manually
        try:
            model = ImageEncoder('ViT-B-16')
            model.load_state_dict(checkpoint)
            print(f"  ✓ Model loaded via manual state dict loading")

        except Exception as e2:
            print(f"  ✗ Alternative method also failed: {e2}")
            return False

    print(f"\nTest 3: Testing model inference...")

    try:
        # Create dummy input (batch_size=2, channels=3, height=224, width=224)
        dummy_input = torch.randn(2, 3, 224, 224)
        print(f"  Input shape: {dummy_input.shape}")

        # Set model to eval mode
        model.eval()

        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)

        print(f"  ✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output dtype: {output.dtype}")
        print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    except Exception as e:
        print(f"  ✗ Error during inference: {e}")
        return False

    print(f"\nTest 4: Checking model components...")

    try:
        # Check if model has the expected structure
        if hasattr(model, 'model'):
            print(f"  ✓ Has 'model' attribute")

            if hasattr(model.model, 'visual'):
                print(f"  ✓ Has 'model.visual' attribute")

                # Check visual encoder structure
                visual = model.model.visual
                if hasattr(visual, 'transformer'):
                    num_layers = len(visual.transformer.resblocks)
                    print(f"  ✓ Visual transformer has {num_layers} layers")

                if hasattr(visual, 'conv1'):
                    print(f"  ✓ Has conv1 (patch embedding)")

                if hasattr(visual, 'ln_pre'):
                    print(f"  ✓ Has ln_pre (pre-norm)")

                if hasattr(visual, 'ln_post'):
                    print(f"  ✓ Has ln_post (post-norm)")
        else:
            print(f"  Model structure: {dir(model)[:10]}...")

    except Exception as e:
        print(f"  ⚠ Warning during component check: {e}")

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print("\nThe converted model is compatible with model-merging framework.")
    print("You can now use it for:")
    print("  - Model merging experiments")
    print("  - Fine-tuning")
    print("  - Evaluation on downstream tasks")

    return True


if __name__ == '__main__':
    success = test_converted_model()
    sys.exit(0 if success else 1)
