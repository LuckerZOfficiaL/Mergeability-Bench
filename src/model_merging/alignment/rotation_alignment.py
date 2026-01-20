"""
Rotation symmetry alignment for CLIP ViT models.
Integrates rotation/permutation alignment from ViT_ImageEncoder_RotationSymmetry.
"""

import sys
import torch
import open_clip
from typing import Dict, Any
import logging
import copy

# Import from ViT_ImageEncoder_RotationSymmetry
sys.path.insert(0, '/home/lzhou/MM/ViT_ImageEncoder_RotationSymmetry')
from src.alignment_utils import (
    extract_attention_params,
    extract_ffn_params,
    chunk_svd_qk,
    chunk_svd_vo,
    permute_ffn,
    update_attention_params,
    update_ffn_params
)


def apply_rotation_alignment(
    finetuned_state_dicts: Dict[Any, Dict[str, torch.Tensor]],
    model_name: str,
    device: str = 'cpu',
    logger: logging.Logger = None
) -> Dict[Any, Dict[str, torch.Tensor]]:
    """
    Apply rotation alignment to all finetuned models.

    Uses first model as anchor, aligns all others to it.
    Verifies weight changes and output invariance for each.

    Args:
        finetuned_state_dicts: Dict mapping dataset configs to state_dicts
        model_name: Model architecture (e.g., 'ViT-B-16')
        device: Device for computation
        logger: Logger for progress messages

    Returns:
        Dict with same keys, values are aligned state_dicts
    """
    if logger:
        logger.info("="*70)
        logger.info("ROTATION SYMMETRY ALIGNMENT")
        logger.info("="*70)

    # Convert dict to list to access by index
    dataset_keys = list(finetuned_state_dicts.keys())

    if len(dataset_keys) < 2:
        if logger:
            logger.info("Only one model provided, no alignment needed.")
        return finetuned_state_dicts

    # Select first model as anchor
    anchor_key = dataset_keys[0]
    anchor_name = anchor_key.name if hasattr(anchor_key, 'name') else str(anchor_key)

    if logger:
        logger.info(f"Selected anchor model: {anchor_name}")
        logger.info(f"Models to align: {len(dataset_keys) - 1}")

    # Load anchor model (this will download model once)
    anchor_state_dict = finetuned_state_dicts[anchor_key]
    anchor_encoder = _state_dict_to_encoder(anchor_state_dict, model_name)

    # Align each model to anchor
    # Return anchor in same format as input (ImageEncoder state dict WITH 'model.' prefix)
    aligned_state_dicts = {anchor_key: anchor_encoder.state_dict()}

    for dataset_key in dataset_keys[1:]:
        dataset_name = dataset_key.name if hasattr(dataset_key, 'name') else str(dataset_key)

        if logger:
            logger.info(f"\nAligning {dataset_name} to {anchor_name}...")

        # Load local model (reuse anchor structure to avoid re-downloading)
        local_state_dict = finetuned_state_dicts[dataset_key]
        local_encoder = _state_dict_to_encoder(local_state_dict, model_name, base_model=anchor_encoder)

        # Keep copy for verification
        original_encoder = copy.deepcopy(local_encoder)

        # Apply alignment (modifies local_encoder in-place)
        align_clip_models(anchor_encoder, local_encoder)

        # Verify alignment
        weights_changed, max_output_diff = verify_alignment(
            original_encoder,
            local_encoder,
            model_name
        )

        if logger:
            if weights_changed:
                logger.info(f"  ✓ Weights changed (alignment applied)")
            else:
                logger.warning(f"  ✗ WARNING: No weights changed (possible bug)")

            if max_output_diff < 1e-4:
                logger.info(f"  ✓ Output invariance verified: max diff = {max_output_diff:.2e}")
            else:
                logger.warning(f"  ✗ WARNING: Output difference too large: {max_output_diff:.2e}")

        # Convert back to state_dict
        # Return ImageEncoder state dict (matches input format WITH 'model.' prefix)
        aligned_state_dicts[dataset_key] = local_encoder.state_dict()

    if logger:
        logger.info(f"\n✓ Rotation alignment complete for {len(dataset_keys) - 1} models!")
        logger.info("="*70)

    return aligned_state_dicts


def align_clip_models(anchor_encoder, local_encoder, num_heads=12, head_size=64):
    """
    Align local_encoder to anchor_encoder using rotation symmetry.
    Modifies local_encoder in-place.

    Implementation ported from align_models.py align_model_to_anchor()

    Args:
        anchor_encoder: ImageEncoder or OpenCLIP model to align to (stays unchanged)
        local_encoder: ImageEncoder or OpenCLIP model to be aligned (modified in-place)
        num_heads: Number of attention heads (12 for ViT-B)
        head_size: Dimension of each head (64 for ViT-B)
    """
    # Access the visual encoder's transformer resblocks
    # Handle both ImageEncoder wrapper and direct OpenCLIP model
    if hasattr(anchor_encoder, 'model'):
        anchor_visual = anchor_encoder.model.visual
        local_visual = local_encoder.model.visual
    else:
        anchor_visual = anchor_encoder.visual
        local_visual = local_encoder.visual

    num_layers = len(local_visual.transformer.resblocks)

    for layer_idx in range(num_layers):
        local_resblock = local_visual.transformer.resblocks[layer_idx]
        anchor_resblock = anchor_visual.transformer.resblocks[layer_idx]

        # ============ Align Attention ============
        # Extract attention parameters
        local_attn_params = extract_attention_params(local_resblock)
        anchor_attn_params = extract_attention_params(anchor_resblock)

        # Align Q, K using rotation (per-head SVD)
        aligned_qk = chunk_svd_qk(
            local_attn_params,
            anchor_attn_params,
            num_heads=num_heads,
            head_size=head_size
        )

        # Align V, O using rotation (per-head SVD)
        aligned_vo = chunk_svd_vo(
            local_attn_params,
            anchor_attn_params,
            num_heads=num_heads,
            head_size=head_size
        )

        # Update attention module with aligned parameters
        update_attention_params(local_resblock, aligned_qk, aligned_vo)

        # ============ Align FFN ============
        # Extract FFN parameters
        local_ffn_params = extract_ffn_params(local_resblock)
        anchor_ffn_params = extract_ffn_params(anchor_resblock)

        # Align FFN using permutation (Hungarian algorithm)
        aligned_ffn = permute_ffn(local_ffn_params, anchor_ffn_params)

        # Update FFN module with aligned parameters
        update_ffn_params(local_resblock, aligned_ffn)


def verify_alignment(
    original_encoder,
    aligned_encoder,
    model_name: str,
    num_tests: int = 5
) -> tuple:
    """
    Verify alignment correctness:
    1. Check that weights changed (some parameters differ)
    2. Check output invariance (outputs identical on random inputs)

    Args:
        original_encoder: Model before alignment
        aligned_encoder: Model after alignment
        model_name: Model architecture name
        num_tests: Number of random inputs to test

    Returns:
        (weights_changed: bool, max_output_diff: float)
    """
    # Check weight changes
    original_state = original_encoder.state_dict()
    aligned_state = aligned_encoder.state_dict()

    num_changed = 0
    for (name, p_orig), (_, p_aligned) in zip(original_state.items(), aligned_state.items()):
        diff = torch.norm(p_orig - p_aligned).item()
        if diff > 1e-6:
            num_changed += 1

    weights_changed = num_changed > 0

    # Check output invariance
    original_encoder.eval()
    aligned_encoder.eval()

    # Access visual encoder
    if hasattr(original_encoder, 'model'):
        original_visual = original_encoder.model.visual
        aligned_visual = aligned_encoder.model.visual
    else:
        original_visual = original_encoder.visual
        aligned_visual = aligned_encoder.visual

    max_diff = 0.0
    with torch.no_grad():
        for _ in range(num_tests):
            # Create random input (batch_size=2, channels=3, height=224, width=224)
            random_input = torch.randn(2, 3, 224, 224)

            # Get visual encoder outputs
            output_orig = original_visual(random_input)
            output_aligned = aligned_visual(random_input)

            # Compute difference
            diff = torch.norm(output_orig - output_aligned).item()
            max_diff = max(max_diff, diff)

    return weights_changed, max_diff


def _state_dict_to_encoder(state_dict: Dict[str, torch.Tensor], model_name: str, base_model=None):
    """
    Convert state_dict to ImageEncoder or OpenCLIP model.

    Args:
        state_dict: PyTorch state dict with model parameters
        model_name: Model architecture (e.g., 'ViT-B-16')
        base_model: Optional base model to clone (avoids re-downloading)

    Returns:
        Model object (either ImageEncoder wrapper or OpenCLIP model)
    """
    # Try to import ImageEncoder
    try:
        from model_merging.model.encoder import ImageEncoder

        # If base_model provided, clone it to avoid re-downloading
        if base_model is not None:
            encoder = copy.deepcopy(base_model)
            encoder.load_state_dict(state_dict)
        else:
            encoder = ImageEncoder(model_name)
            encoder.load_state_dict(state_dict)
        return encoder
    except Exception as e:
        # Fallback to direct OpenCLIP
        print(f"WARNING: Failed to use ImageEncoder, falling back to OpenCLIP: {e}")
        if base_model is not None:
            model = copy.deepcopy(base_model)
            model.load_state_dict(state_dict, strict=False)
        else:
            model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained='openai')
            model.load_state_dict(state_dict, strict=False)
        return model
