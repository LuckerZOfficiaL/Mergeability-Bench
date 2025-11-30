"""
High-level interface for performing permutation matching on task vectors.
"""

import logging
from typing import Dict

import torch
from torch import Tensor

from model_merging.permutations.permutation_spec import PermutationSpec, AutoPermutationSpecBuilder
from model_merging.perm_matching.weight_matching import weight_matching_for_task_vectors, LayerIterationOrder
from model_merging.perm_matching.utils import apply_permutation_to_statedict

pylogger = logging.getLogger(__name__)


def _get_permutation_spec_for_model(model_name: str):
    """
    Get the appropriate permutation specification builder for a given model.

    Args:
        model_name: Name of the model architecture

    Returns:
        PermutationSpec or None if not supported
    """
    from model_merging.permutations.permutation_spec import (
        ViTPermutationSpecBuilder,
        ResNet20PermutationSpecBuilder,
        ResNet50PermutationSpecBuilder,
        VGG16PermutationSpecBuilder,
        MLPPermutationSpecBuilder,
    )
    from model_merging.perm_matching.clip_vit_spec import create_clip_vit_spec

    model_name_lower = model_name.lower()

    # Check for CLIP ViT models first (most common in your setup)
    # CLIP models: ViT-B-32, ViT-B-16, ViT-L-14, etc.
    if "vit-" in model_name_lower or "vit_" in model_name_lower:
        pylogger.info(f"Detected CLIP ViT model: {model_name}")
        return create_clip_vit_spec(model_name)

    # Standard ViT models (timm/torchvision style)
    elif "vit" in model_name_lower or "vision_transformer" in model_name_lower:
        # ViT models - need to determine depth
        # Common ViT variants: vit_b_16 (12 layers), vit_l_16 (24 layers)
        if "_b_" in model_name_lower or "base" in model_name_lower:
            depth = 12
        elif "_l_" in model_name_lower or "large" in model_name_lower:
            depth = 24
        elif "_s_" in model_name_lower or "small" in model_name_lower:
            depth = 12
        else:
            pylogger.warning(f"Unknown ViT variant: {model_name}, assuming depth=12")
            depth = 12

        pylogger.info(f"Creating standard ViT permutation spec with depth={depth}")
        builder = ViTPermutationSpecBuilder(depth=depth)
        return builder.create_permutation_spec()

    elif "resnet20" in model_name_lower:
        pylogger.info("Creating ResNet20 permutation spec")
        builder = ResNet20PermutationSpecBuilder()
        return builder.create_permutation_spec()

    elif "resnet50" in model_name_lower or "resnet-50" in model_name_lower:
        pylogger.info("Creating ResNet50 permutation spec")
        builder = ResNet50PermutationSpecBuilder()
        return builder.create_permutation()

    elif "vgg16" in model_name_lower or "vgg-16" in model_name_lower:
        pylogger.info("Creating VGG16 permutation spec")
        builder = VGG16PermutationSpecBuilder()
        return builder.create_permutation()

    else:
        return None


def apply_permutation_to_task_vectors(
    pretrained_state_dict: Dict[str, Tensor],
    task_dicts: Dict[str, Dict[str, Tensor]],
    model_name: str,
    max_iter: int = 100,
    verbose: bool = False,
) -> Dict[str, Dict[str, Tensor]]:
    """
    Apply permutation matching to align task vectors before computing mergeability metrics.

    This function takes multiple task vectors and finds permutations to align them all
    to a reference task vector (the first one in the dictionary).

    Args:
        pretrained_state_dict: State dict of the pretrained model
        task_dicts: Dictionary mapping dataset names to task vectors (param dicts)
        model_name: Name of the model (for creating permutation spec)
        max_iter: Maximum iterations for weight matching
        verbose: Whether to print detailed progress

    Returns:
        Dictionary of aligned task vectors (same structure as input task_dicts)
    """
    if len(task_dicts) < 2:
        pylogger.info("Only one task vector provided, no permutation matching needed")
        return task_dicts

    # Create permutation specification
    pylogger.info(f"Creating permutation specification for model: {model_name}")

    try:
        perm_spec = _get_permutation_spec_for_model(model_name)

        if perm_spec is None:
            pylogger.warning(
                f"Permutation matching is not supported for model architecture: {model_name}. "
                f"Supported architectures: ViT, ResNet20, ResNet50, VGG16. "
                f"Returning original task vectors without permutation matching."
            )
            return task_dicts

        # Verify that the permutation spec actually matches the model
        # by checking if any parameters in the task vectors are in the spec
        dataset_names = list(task_dicts.keys())
        first_task = task_dicts[dataset_names[0]]

        matching_params = [
            param_name for param_name in first_task.keys()
            if param_name in perm_spec.layer_and_axes_to_perm
        ]

        if len(matching_params) == 0:
            pylogger.warning(
                f"Permutation spec was created for {model_name}, but no parameter names match. "
                f"This usually means the model uses a different naming convention than expected. "
                f"Sample parameter names: {list(first_task.keys())[:5]}... "
                f"Returning original task vectors without permutation matching."
            )
            return task_dicts

        pylogger.info(f"Permutation spec validated: {len(matching_params)}/{len(first_task)} parameters match")

    except Exception as e:
        pylogger.error(f"Failed to create permutation spec: {e}")
        pylogger.info("Returning original task vectors without permutation matching")
        return task_dicts

    # Select first task as the reference (fixed)
    dataset_names = list(task_dicts.keys())
    reference_dataset = dataset_names[0]

    pylogger.info(f"Using {reference_dataset} as reference for permutation matching")

    # Reconstruct fine-tuned state dicts from task vectors
    def task_vector_to_finetuned(task_dict):
        return {k: pretrained_state_dict[k] + task_dict[k] for k in task_dict.keys()}

    reference_finetuned = task_vector_to_finetuned(task_dicts[reference_dataset])

    # Apply permutation matching to each other task vector
    aligned_task_dicts = {reference_dataset: task_dicts[reference_dataset]}

    for dataset_name in dataset_names[1:]:
        pylogger.info(f"Aligning {dataset_name} to {reference_dataset}...")

        try:
            permutee_finetuned = task_vector_to_finetuned(task_dicts[dataset_name])

            # Find optimal permutation
            perm_indices = weight_matching_for_task_vectors(
                ps=perm_spec,
                pretrained_params=pretrained_state_dict,
                fixed_finetuned_params=reference_finetuned,
                permutee_finetuned_params=permutee_finetuned,
                max_iter=max_iter,
                layer_iteration_order=LayerIterationOrder.RANDOM,
                verbose=verbose,
            )

            # Apply permutation to the fine-tuned model
            permuted_finetuned = apply_permutation_to_statedict(
                perm_spec, perm_indices, permutee_finetuned
            )

            # Compute aligned task vector
            aligned_task_vector = {
                k: permuted_finetuned[k] - pretrained_state_dict[k]
                for k in permutee_finetuned.keys()
                if k in pretrained_state_dict
            }

            aligned_task_dicts[dataset_name] = aligned_task_vector
            pylogger.info(f"Finished aligning {dataset_name}")

        except Exception as e:
            pylogger.error(f"Failed to align {dataset_name}: {e}")
            pylogger.info(f"Using original task vector for {dataset_name}")
            aligned_task_dicts[dataset_name] = task_dicts[dataset_name]

    return aligned_task_dicts
