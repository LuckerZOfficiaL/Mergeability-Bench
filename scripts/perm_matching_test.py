"""
Simple test to verify TransFusion's permutation matching works.

This script:
1. Loads two ViT-B-16 models (MNIST and Cars)
2. Applies permutation matching from TransFusion
3. Checks that model B and permuted model B have different weights
4. Checks that both produce the same output (output invariance)

Usage:
    python scripts/perm_matching_test.py
"""

import sys
import torch
from pathlib import Path

# Add TransFusion to path
sys.path.insert(0, '/home/ubuntu/thesis/MM/TransFusion')

# Add model-merging to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from TransFusion
import open_clip
from src.models import OpenCLIPModel
from permutations.weights_matcher import WeightMatcher
from permutations.permutation_spec import CLIP_Visual_PermutationSpecBuilder
from permutations.utils import apply_permutation_to_statedict

# Import from model-merging
from model_merging.utils.io_utils import load_model_from_hf


def main():
    print("=" * 80)
    print("SIMPLE PERMUTATION MATCHING TEST")
    print("=" * 80)

    # Step 1: Load two models from model-merging
    print("\n[Step 1] Loading models from model-merging...")
    model_a_wrapper = load_model_from_hf("ViT-B-16", "MNIST")
    model_b_wrapper = load_model_from_hf("ViT-B-16", "Cars")

    # Get the underlying CLIP models
    model_a_sd_raw = model_a_wrapper.state_dict()
    model_b_sd_raw = model_b_wrapper.state_dict()

    # Remove 'model.' prefix from state dict keys
    model_a_sd = {k.replace('model.', ''): v for k, v in model_a_sd_raw.items()}
    model_b_sd = {k.replace('model.', ''): v for k, v in model_b_sd_raw.items()}

    print(f"  Model A (MNIST) loaded: {len(model_a_sd)} parameters")
    print(f"  Model B (Cars) loaded: {len(model_b_sd)} parameters")

    # Step 2: Apply TransFusion's setup_visual transformation
    print("\n[Step 2] Applying TransFusion's setup_visual() transformation...")

    # Create OpenCLIP models
    clip_model_a, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='openai', cache_dir='/tmp/open_clip'
    )
    clip_model_b, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-16', pretrained='openai', cache_dir='/tmp/open_clip'
    )

    # Load our fine-tuned weights (strict=False because we only have visual encoder)
    clip_model_a.load_state_dict(model_a_sd, strict=False)
    clip_model_b.load_state_dict(model_b_sd, strict=False)

    # Apply TransFusion's transformation
    wrapped_a = OpenCLIPModel(clip_model_a)
    wrapped_b = OpenCLIPModel(clip_model_b)

    visual_a = wrapped_a.clip_model.visual
    visual_b = wrapped_b.clip_model.visual

    print("  TransFusion transformation applied (Q/K/V split, MLP renamed)")

    # Step 3: Run permutation matching
    print("\n[Step 3] Running TransFusion's permutation matching...")

    # Build permutation spec
    ps_builder = CLIP_Visual_PermutationSpecBuilder(depth=12)
    permutation_spec = ps_builder.create_permutation_spec()

    # Get state dicts and move to CPU for matching
    state_dict_a = {k: v.cpu() for k, v in visual_a.state_dict().items()}
    state_dict_b = {k: v.cpu() for k, v in visual_b.state_dict().items()}

    # Run weight matcher
    weight_matcher = WeightMatcher(
        ps=permutation_spec,
        fixed=state_dict_a,
        permutee=state_dict_b,
        num_heads=12,
        intra_head=True,
        max_iter=10,
    )

    perm_indices, heads_perm = weight_matcher.run()
    print(f"  Permutation matching completed")
    print(f"  Number of permutation groups: {len(perm_indices)}")

    # Check how many are identity
    num_identity = sum(
        1 for perm in perm_indices.values()
        if torch.equal(perm, torch.arange(len(perm)))
    )
    print(f"  Identity permutations: {num_identity}/{len(perm_indices)}")
    print(f"  Non-identity permutations: {len(perm_indices) - num_identity}/{len(perm_indices)}")

    # Step 4: Apply permutation
    print("\n[Step 4] Applying permutation to model B...")
    permuted_state_dict_b = apply_permutation_to_statedict(
        permutation_spec, perm_indices, state_dict_b, heads_perm
    )
    visual_b.load_state_dict(permuted_state_dict_b)
    print("  Permutation applied")

    # Step 5: Check that weights are different
    print("\n[Step 5] Checking that model B weights changed...")
    num_different = 0
    num_same = 0

    for key in state_dict_b.keys():
        if not torch.equal(state_dict_b[key], permuted_state_dict_b[key]):
            num_different += 1
        else:
            num_same += 1

    print(f"  Parameters changed: {num_different}/{len(state_dict_b)}")
    print(f"  Parameters unchanged: {num_same}/{len(state_dict_b)}")

    if num_different > 0:
        print("  ✓ Weights are different!")
    else:
        print("  ✗ WARNING: All weights are the same (identity permutation)")

    # Step 6: Check output invariance
    print("\n[Step 6] Checking output invariance...")

    # Create random input (batch_size=4, channels=3, height=224, width=224)
    random_input = torch.randn(4, 3, 224, 224)

    # Get outputs from original model B
    with torch.no_grad():
        # Need to reload original model B
        clip_model_b_orig, _, _ = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained='openai', cache_dir='/tmp/open_clip'
        )
        clip_model_b_orig.load_state_dict(model_b_sd, strict=False)
        wrapped_b_orig = OpenCLIPModel(clip_model_b_orig)
        visual_b_orig = wrapped_b_orig.clip_model.visual

        output_original = visual_b_orig(random_input)
        output_permuted = visual_b(random_input)

    # Compare outputs
    max_diff = torch.abs(output_original - output_permuted).max().item()
    mean_diff = torch.abs(output_original - output_permuted).mean().item()

    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-4:
        print("  ✓ Output invariance verified!")
    else:
        print("  ✗ WARNING: Outputs are different!")

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
