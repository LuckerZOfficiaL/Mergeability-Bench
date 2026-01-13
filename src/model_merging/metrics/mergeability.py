"""
Mergeability metrics for predicting model merging outcomes.

This module provides various metrics to measure the compatibility/similarity
between two task vectors, which can be used to predict the success of model merging.
"""

import math
import copy
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


def flatten_task_dict(task_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten all tensors in a task dict into a single 1D vector.

    Args:
        task_dict: Dictionary mapping layer names to parameter tensors.

    Returns:
        A single flattened tensor containing all parameters.
    """
    tensors = []
    for key in sorted(task_dict.keys()):
        tensor = task_dict[key]
        if tensor.dtype in [torch.int64, torch.uint8]:
            continue
        tensors.append(tensor.flatten().float())
    return torch.cat(tensors)


def get_layer_vectors(
    task_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Extract flattened vectors for each layer.

    Args:
        task_dict: Dictionary mapping layer names to parameter tensors.

    Returns:
        Dictionary mapping layer names to flattened tensors.
    """
    result = {}
    for key, tensor in task_dict.items():
        if tensor.dtype in [torch.int64, torch.uint8]:
            continue
        result[key] = tensor.flatten().float()
    return result


# =============================================================================
# Per-Layer Computation Wrapper
# =============================================================================


def compute_metric_per_layer(
    metric_fn: Callable,
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute any metric for each layer separately.

    Args:
        metric_fn: A metric function that takes two task dicts and returns a float.
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Dictionary mapping layer names to metric values.
    """
    layers_1 = get_layer_vectors(task_dict_1)
    layers_2 = get_layer_vectors(task_dict_2)

    result = {}
    common_keys = set(layers_1.keys()) & set(layers_2.keys())

    for key in sorted(common_keys):
        # Create single-layer task dicts
        layer_dict_1 = {key: task_dict_1[key]}
        layer_dict_2 = {key: task_dict_2[key]}
        try:
            result[key] = metric_fn(layer_dict_1, layer_dict_2)
        except Exception:
            result[key] = float('nan')

    return result


def compute_metric_layer_wise_avg(
    metric_fn: Callable,
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute average of a metric across all layers.

    Args:
        metric_fn: A metric function that takes two task dicts and returns a float.
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Average metric value across layers.
    """
    per_layer = compute_metric_per_layer(metric_fn, task_dict_1, task_dict_2)
    valid_values = [v for v in per_layer.values() if not math.isnan(v)]
    if not valid_values:
        return 0.0
    return sum(valid_values) / len(valid_values)


# =============================================================================
# Core Metrics
# =============================================================================


def task_vector_cosine_similarity(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute cosine similarity between two task vectors.

    This is one of the most intuitive metrics: if two task vectors point
    in similar directions in weight space, they may be more compatible.

    Args:
        task_dict_1: First task vector (finetuned - pretrained).
        task_dict_2: Second task vector.

    Returns:
        Cosine similarity value in [-1, 1].
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()


def task_vector_l2_distance(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute L2 (Euclidean) distance between two task vectors.

    Measures how far apart the two task vectors are in weight space.
    Smaller distance might indicate more compatible tasks.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        L2 distance (non-negative).
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    return torch.norm(vec1 - vec2, p=2).item()


def task_vector_dot_product(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute dot product between two task vectors.

    Unlike cosine similarity, this is not normalized by magnitude,
    so it captures both direction and magnitude information.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Dot product value.
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    return torch.dot(vec1, vec2).item()


def weight_space_angle(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute angle between two task vectors in weight space (in degrees).

    This is derived from cosine similarity but expressed as an angle,
    which can be more intuitive for interpretation.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Angle in degrees [0, 180].
    """
    cos_sim = task_vector_cosine_similarity(task_dict_1, task_dict_2)
    # Clamp to handle numerical errors
    cos_sim = max(-1.0, min(1.0, cos_sim))
    angle_rad = math.acos(cos_sim)
    return math.degrees(angle_rad)


def task_vector_magnitude_ratio(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute ratio of task vector magnitudes (smaller / larger).

    If one task vector is much larger than the other, the smaller task
    might get "overwhelmed" during merging. A ratio close to 1 suggests
    more balanced contributions.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Magnitude ratio in (0, 1].
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    mag1 = torch.norm(vec1, p=2).item()
    mag2 = torch.norm(vec2, p=2).item()

    if mag1 < 1e-8 or mag2 < 1e-8:
        return 0.0

    return min(mag1, mag2) / max(mag1, mag2)


# =============================================================================
# SVD-based Metrics
# =============================================================================


def effective_rank(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute effective rank (participation ratio) of the two task vectors.

    The effective rank measures the intrinsic dimensionality of the subspace
    spanned by the two task vectors. It is computed using the entropy of the
    normalized singular value distribution.

    Effective Rank = exp(H(p)) where H(p) = -Σ p_i log(p_i)
    and p_i = σ_i / Σσ_j (normalized singular values)

    Interpretation:
        - Effective rank ≈ 1.0: Task vectors are highly aligned (excellent mergeability)
        - Effective rank ≈ 1.5: Moderate alignment (good mergeability)
        - Effective rank ≈ 2.0: Task vectors are orthogonal (poor mergeability)

    This metric is based on the hypothesis that models lying in the linear
    tangent space of the pretrained model should have aligned task vectors,
    resulting in low effective rank.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Effective rank value in [1, 2].
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    # Stack as matrix (2 × D)
    task_matrix = torch.stack([vec1, vec2], dim=0)

    # Compute SVD
    try:
        _, S, _ = torch.linalg.svd(task_matrix, full_matrices=False)
    except Exception:
        return 2.0  # Return worst case on failure

    # Normalize singular values to form probability distribution
    S_normalized = S / (S.sum() + 1e-10)

    # Compute entropy
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()

    # Effective rank
    eff_rank = torch.exp(entropy).item()

    return eff_rank


def effective_rank_mergeability_score(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute mergeability score from effective rank (mapped to [0, 1]).

    This is a normalized version of effective_rank where:
        - Score = 1.0 means perfect alignment (effective rank = 1.0)
        - Score = 0.0 means orthogonal (effective rank = 2.0)

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Mergeability score in [0, 1], where higher is better.
    """
    eff_rank = effective_rank(task_dict_1, task_dict_2)

    # Map [1, 2] to [1, 0]
    score = 2.0 - eff_rank
    score = max(0.0, min(1.0, score))

    return score


def stable_rank(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute stable rank of the two task vectors.

    Stable rank is an alternative measure of effective dimensionality:
    Stable Rank = (Σσ_i)² / Σσ_i²

    This is related to effective rank but uses L2 norm instead of entropy.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Stable rank value in [1, 2].
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    # Stack as matrix (2 × D)
    task_matrix = torch.stack([vec1, vec2], dim=0)

    # Compute SVD
    try:
        _, S, _ = torch.linalg.svd(task_matrix, full_matrices=False)
    except Exception:
        return 2.0

    # Stable rank = (sum of singular values)^2 / sum of squared singular values
    s_rank = (S.sum() ** 2) / ((S ** 2).sum() + 1e-10)

    return s_rank.item()


def spectral_gap(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute spectral gap between the two largest singular values.

    The spectral gap measures the difference between the first and second
    singular values, normalized by the first. A large gap indicates strong
    alignment (one dominant direction).

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Spectral gap in [0, 1], where larger means better alignment.
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    # Stack as matrix (2 × D)
    task_matrix = torch.stack([vec1, vec2], dim=0)

    # Compute SVD
    try:
        _, S, _ = torch.linalg.svd(task_matrix, full_matrices=False)
    except Exception:
        return 0.0

    if len(S) < 2:
        return 1.0  # Only one singular value means perfect alignment

    # Spectral gap = (σ_1 - σ_2) / σ_1
    gap = (S[0] - S[1]) / (S[0] + 1e-10)

    return gap.item()


def singular_value_ratio(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute ratio of second to first singular value.

    This is complementary to spectral_gap. A small ratio indicates
    strong alignment (second direction is weak).

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Ratio in [0, 1], where smaller means better alignment.
    """
    vec1 = flatten_task_dict(task_dict_1)
    vec2 = flatten_task_dict(task_dict_2)

    # Stack as matrix (2 × D)
    task_matrix = torch.stack([vec1, vec2], dim=0)

    # Compute SVD
    try:
        _, S, _ = torch.linalg.svd(task_matrix, full_matrices=False)
    except Exception:
        return 1.0

    if len(S) < 2:
        return 0.0  # Only one singular value means perfect alignment

    # Ratio = σ_2 / σ_1
    ratio = S[1] / (S[0] + 1e-10)

    return ratio.item()


def layerwise_effective_rank(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute weighted average effective rank across all layers.

    This computes the effective rank for each layer separately, then
    takes a weighted average based on the magnitude of updates in each layer.
    This provides more granular insight than global effective rank.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Weighted average effective rank across layers.
    """
    layers_1 = get_layer_vectors(task_dict_1)
    layers_2 = get_layer_vectors(task_dict_2)

    common_keys = set(layers_1.keys()) & set(layers_2.keys())

    layer_ranks = []
    layer_weights = []

    for key in sorted(common_keys):
        delta_A = layers_1[key]
        delta_B = layers_2[key]

        # Skip if no updates
        if delta_A.norm() < 1e-10 or delta_B.norm() < 1e-10:
            continue

        # Stack and compute SVD
        layer_matrix = torch.stack([delta_A, delta_B])

        try:
            _, S, _ = torch.linalg.svd(layer_matrix, full_matrices=False)
        except Exception:
            continue

        # Effective rank for this layer
        S_norm = S / (S.sum() + 1e-10)
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum()
        eff_rank = torch.exp(entropy).item()

        # Weight by total update magnitude
        weight = (delta_A.norm() + delta_B.norm()).item()

        layer_ranks.append(eff_rank)
        layer_weights.append(weight)

    if not layer_ranks:
        return 2.0  # Return worst case if no valid layers

    # Weighted average
    total_weight = sum(layer_weights)
    weighted_avg = sum(r * w for r, w in zip(layer_ranks, layer_weights)) / total_weight

    return weighted_avg


def layerwise_effective_rank_mergeability_score(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
) -> float:
    """Compute mergeability score from layerwise effective rank (mapped to [0, 1]).

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.

    Returns:
        Mergeability score in [0, 1], where higher is better.
    """
    eff_rank = layerwise_effective_rank(task_dict_1, task_dict_2)

    # Map [1, 2] to [1, 0]
    score = 2.0 - eff_rank
    score = max(0.0, min(1.0, score))

    return score


def singular_value_overlap(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 100,
) -> float:
    """Compute overlap of top-k singular values across weight matrices.

    This measures whether the two task vectors modify similar "directions"
    of the weight matrices in terms of their singular value structure.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.
        top_k: Number of top singular values to consider per matrix.

    Returns:
        Average overlap coefficient across all 2D weight matrices.
    """
    overlaps = []

    for key in sorted(task_dict_1.keys()):
        if key not in task_dict_2:
            continue

        tensor1 = task_dict_1[key]
        tensor2 = task_dict_2[key]

        # Only process 2D matrices
        if tensor1.dim() != 2:
            continue

        # Compute SVD for both
        try:
            _, s1, _ = torch.linalg.svd(tensor1.float(), full_matrices=False)
            _, s2, _ = torch.linalg.svd(tensor2.float(), full_matrices=False)
        except Exception:
            continue

        # Normalize singular values
        s1 = s1[:top_k] / (s1.sum() + 1e-8)
        s2 = s2[:top_k] / (s2.sum() + 1e-8)

        # Pad to same length if needed
        max_len = max(len(s1), len(s2))
        if len(s1) < max_len:
            s1 = F.pad(s1, (0, max_len - len(s1)))
        if len(s2) < max_len:
            s2 = F.pad(s2, (0, max_len - len(s2)))

        # Compute overlap as cosine similarity of normalized singular value distributions
        overlap = F.cosine_similarity(s1.unsqueeze(0), s2.unsqueeze(0)).item()
        overlaps.append(overlap)

    if not overlaps:
        return 0.0

    return sum(overlaps) / len(overlaps)


def subspace_overlap(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 10,
) -> float:
    """Compute principal left subspace overlap between task vectors.

    Measures how much the principal left directions (from SVD, using U matrices)
    of the two task vectors overlap. High overlap might indicate task compatibility.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.
        top_k: Number of top principal directions to consider.

    Returns:
        Average left subspace overlap across all 2D weight matrices.
    """
    overlaps = []

    for key in sorted(task_dict_1.keys()):
        if key not in task_dict_2:
            continue

        tensor1 = task_dict_1[key]
        tensor2 = task_dict_2[key]

        # Only process 2D matrices
        if tensor1.dim() != 2:
            continue

        # Compute SVD for both
        try:
            u1, _, _ = torch.linalg.svd(tensor1.float(), full_matrices=False)
            u2, _, _ = torch.linalg.svd(tensor2.float(), full_matrices=False)
        except Exception:
            continue

        # Take top-k columns of U matrices
        k = min(top_k, u1.shape[1], u2.shape[1])
        u1_k = u1[:, :k]
        u2_k = u2[:, :k]

        # Compute subspace overlap using Frobenius norm of U1^T @ U2
        # Maximum overlap is sqrt(k) when subspaces are identical
        product = u1_k.T @ u2_k
        overlap = torch.norm(product, p='fro').item() / k
        overlaps.append(overlap)

    if not overlaps:
        return 0.0

    return sum(overlaps) / len(overlaps)


def right_subspace_overlap(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 10,
) -> Tuple[float, float]:
    """Compute principal right subspace overlap between task vectors for both top-k and bottom-k.

    Measures how much the principal right directions (from SVD, using V matrices)
    of the two task vectors overlap. High overlap might indicate task compatibility.
    This metric computes overlap for both the strongest (top-k) and weakest (bottom-k)
    singular vectors.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.
        top_k: Number of principal directions to consider from top and bottom.

    Returns:
        Tuple of (top_k_overlap, bottom_k_overlap):
        - top_k_overlap: Average right subspace overlap using top-k singular vectors
        - bottom_k_overlap: Average right subspace overlap using bottom-k singular vectors
    """
    top_overlaps = []
    bottom_overlaps = []

    for key in sorted(task_dict_1.keys()):
        if key not in task_dict_2:
            continue

        tensor1 = task_dict_1[key]
        tensor2 = task_dict_2[key]

        # Only process 2D matrices
        if tensor1.dim() != 2:
            continue

        # Compute SVD for both
        try:
            _, s1, v1 = torch.linalg.svd(tensor1.float(), full_matrices=False)
            _, s2, v2 = torch.linalg.svd(tensor2.float(), full_matrices=False)
        except Exception:
            continue

        # Determine k for this layer
        k = min(top_k, v1.shape[0], v2.shape[0])

        # Top-k: Take first k rows of V matrices (V is returned as V^H in torch.linalg.svd)
        v1_top_k = v1[:k, :]
        v2_top_k = v2[:k, :]

        # Compute top-k subspace overlap using Frobenius norm of V1 @ V2^T
        # Maximum overlap is sqrt(k) when subspaces are identical
        product_top = v1_top_k @ v2_top_k.T
        overlap_top = torch.norm(product_top, p='fro').item() / (k ** 0.5)
        top_overlaps.append(overlap_top)

        # Bottom-k: Take last k rows of V matrices (weakest singular vectors)
        v1_bottom_k = v1[-k:, :]
        v2_bottom_k = v2[-k:, :]

        # Compute bottom-k subspace overlap
        product_bottom = v1_bottom_k @ v2_bottom_k.T
        overlap_bottom = torch.norm(product_bottom, p='fro').item() / (k ** 0.5)
        bottom_overlaps.append(overlap_bottom)

    if not top_overlaps:
        return 0.0, 0.0

    avg_top = sum(top_overlaps) / len(top_overlaps)
    avg_bottom = sum(bottom_overlaps) / len(bottom_overlaps)

    return avg_top, avg_bottom


def interaction_matrix_overlap(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 10,
) -> Tuple[float, float]:
    """Compute interaction matrix overlap between task vectors for both top-k and bottom-k.

    For each layer, computes the interaction matrix M = V_A^T @ V_B where V_A and V_B
    are the right singular vectors. The singular values of M represent the cosines of
    principal angles between the subspaces. The metric returns the average of squared
    singular values, computed separately for top-k and bottom-k singular vectors.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.
        top_k: Number of principal directions to consider from top and bottom.

    Returns:
        Tuple of (top_k_overlap, bottom_k_overlap):
        - top_k_overlap: Average of squared singular values using top-k singular vectors
        - bottom_k_overlap: Average of squared singular values using bottom-k singular vectors
    """
    top_overlaps = []
    bottom_overlaps = []

    for key in sorted(task_dict_1.keys()):
        if key not in task_dict_2:
            continue

        tensor1 = task_dict_1[key]
        tensor2 = task_dict_2[key]

        # Only process 2D matrices
        if tensor1.dim() != 2:
            continue

        # Compute SVD for both
        try:
            _, s1, v1 = torch.linalg.svd(tensor1.float(), full_matrices=False)
            _, s2, v2 = torch.linalg.svd(tensor2.float(), full_matrices=False)
        except Exception:
            continue

        # Determine k for this layer
        k = min(top_k, v1.shape[0], v2.shape[0])

        # Top-k: Take first k rows of V matrices
        v1_top_k = v1[:k, :]
        v2_top_k = v2[:k, :]

        # Compute interaction matrix M = V_A^T @ V_B
        # v1_top_k has shape (k, n), v2_top_k has shape (k, n)
        # M = v1_top_k @ v2_top_k.T has shape (k, k)
        interaction_matrix_top = v1_top_k @ v2_top_k.T

        # Compute SVD on interaction matrix to get singular values
        try:
            _, sigma_top, _ = torch.linalg.svd(interaction_matrix_top, full_matrices=False)
        except Exception:
            continue

        # Average of squared singular values
        overlap_top = torch.mean(sigma_top ** 2).item()
        top_overlaps.append(overlap_top)

        # Bottom-k: Take last k rows of V matrices (weakest singular vectors)
        v1_bottom_k = v1[-k:, :]
        v2_bottom_k = v2[-k:, :]

        # Compute interaction matrix for bottom-k
        interaction_matrix_bottom = v1_bottom_k @ v2_bottom_k.T

        # Compute SVD on interaction matrix
        try:
            _, sigma_bottom, _ = torch.linalg.svd(interaction_matrix_bottom, full_matrices=False)
        except Exception:
            continue

        # Average of squared singular values
        overlap_bottom = torch.mean(sigma_bottom ** 2).item()
        bottom_overlaps.append(overlap_bottom)

    if not top_overlaps:
        return 0.0, 0.0

    avg_top = sum(top_overlaps) / len(top_overlaps)
    avg_bottom = sum(bottom_overlaps) / len(bottom_overlaps)

    return avg_top, avg_bottom


# =============================================================================
# Activation-Based Metrics Infrastructure
# =============================================================================


def build_calibration_loader(
    dataset_configs: List[Dict],
    pretrained_encoder,
    n_samples: int = 10,
    batch_size: int = 32,
    device: str = "cuda",
    random_seed: int = 42,
) -> DataLoader:
    """Build a calibration data loader from multiple datasets.

    Args:
        dataset_configs: List of dataset config dictionaries (from Hydra)
        pretrained_encoder: The pretrained encoder model to get preprocessor
        n_samples: Number of samples to take from each dataset's validation set
        batch_size: Batch size for the calibration loader
        device: Device to use
        random_seed: Random seed for reproducible sampling

    Returns:
        DataLoader containing calibration samples from all datasets
    """
    from model_merging.data.dataset import load_dataset
    from hydra.utils import instantiate
    import random

    all_samples = []
    preprocess_fn = pretrained_encoder.val_preprocess

    # Set random seed for reproducibility
    random.seed(random_seed)

    for dataset_cfg in dataset_configs:
        try:
            # Instantiate the HF dataset using Hydra
            hf_dataset = instantiate(dataset_cfg.hf_dataset)

            # Load the dataset
            dataset = load_dataset(
                name=dataset_cfg.name,
                hf_dataset=hf_dataset,
                preprocess_fn=preprocess_fn,
                ft_epochs=dataset_cfg.get("ft_epochs", 10),
                split_map=dataset_cfg.get("split_map", None),
                batch_size=batch_size,
                label_map=dataset_cfg.get("label_map", None),
                classnames_override=dataset_cfg.get("classnames_override", None),
            )

            # Sample n_samples from validation/test set randomly
            test_dataset = dataset.test_dataset
            n_available = len(test_dataset)
            n_to_sample = min(n_samples, n_available)

            # Random sampling with fixed seed for reproducibility
            indices = random.sample(range(n_available), n_to_sample)
            indices.sort()  # Sort for consistent ordering

            for idx in indices:
                all_samples.append(test_dataset[idx])

        except Exception as e:
            print(f"Warning: Failed to load dataset {dataset_cfg.get('name', 'unknown')}: {e}")
            continue

    if not all_samples:
        raise ValueError("No samples could be loaded from any dataset")

    # Create a simple Dataset wrapper
    class CalibrationDataset(torch.utils.data.Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.samples[idx]

    calibration_dataset = CalibrationDataset(all_samples)
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return calibration_loader


def reconstruct_model_from_task_dict(
    pretrained_model,
    task_dict: Dict[str, torch.Tensor],
    coefficient: float = 1.0,
    device: str = "cuda",
):
    """Reconstruct a finetuned model from pretrained model + task vector.

    Args:
        pretrained_model: The pretrained model
        task_dict: Task vector (finetuned - pretrained weights)
        coefficient: Scaling coefficient for task vector
        device: Device to use

    Returns:
        Reconstructed model on the specified device
    """
    # Import here to avoid circular imports
    from model_merging.utils.utils import apply_dict_to_model

    # Create a deep copy of the pretrained model
    model = copy.deepcopy(pretrained_model)
    model = model.to(device)

    # Apply task vector to get finetuned model
    model = apply_dict_to_model(task_dict, model, coefficient=coefficient)
    model.eval()

    return model


def extract_layer_activations(
    model,
    calibration_loader: DataLoader,
    layer_name: str,
    device: str = "cuda",
) -> torch.Tensor:
    """Extract and average activations from a specific layer over calibration data.

    Args:
        model: The model to extract activations from
        calibration_loader: DataLoader containing calibration samples
        layer_name: Name of the layer to extract activations from
        device: Device to use

    Returns:
        Averaged activation tensor across all calibration samples
    """
    # Import here to avoid circular imports
    from model_merging.utils.utils import get_hook_fn

    model = model.to(device)
    model.eval()

    # Initialize storage for intermediate features
    model.middle_features = {}

    # Find the target module
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break

    if target_module is None:
        raise ValueError(f"Layer {layer_name} not found in model")

    # Register hook
    hook_fn = get_hook_fn(model, layer_name, input_or_output="output")
    handle = target_module.register_forward_hook(hook_fn)

    # Collect activations
    all_activations = []

    with torch.no_grad():
        for batch in calibration_loader:
            images, _ = batch
            images = images.to(device)

            # Forward pass
            _ = model(images)

            # Get activation from this batch
            activation = model.middle_features[layer_name]

            # Average over batch dimension and flatten spatial dimensions if needed
            # Shape is typically (B, seq_len, hidden_dim) or (B, hidden_dim)
            if activation.dim() == 3:
                # Average over sequence dimension: (B, seq_len, hidden_dim) -> (B, hidden_dim)
                activation = activation.mean(dim=1)

            all_activations.append(activation.cpu())

    # Remove hook
    handle.remove()

    # Concatenate all batches and compute average
    all_activations = torch.cat(all_activations, dim=0)  # (N_total, hidden_dim)
    avg_activation = all_activations.mean(dim=0)  # (hidden_dim,)

    return avg_activation


# =============================================================================
# Activation-Based Metrics
# =============================================================================


def activation_l2_distance(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    pretrained_model=None,
    calibration_loader: Optional[DataLoader] = None,
    layer_name: Optional[str] = None,
    device: str = "cuda",
) -> float:
    """Compute L2 distance between average activations of two models.

    This measures how different the internal representations are between
    two finetuned models on the same calibration data.

    Args:
        task_dict_1: First task vector
        task_dict_2: Second task vector
        pretrained_model: Pretrained model (required for activation metrics)
        calibration_loader: DataLoader with calibration samples (required)
        layer_name: Name of layer to extract activations from (required)
        device: Device to use

    Returns:
        L2 distance between averaged activations
    """
    if pretrained_model is None or calibration_loader is None or layer_name is None:
        raise ValueError(
            "Activation metrics require pretrained_model, calibration_loader, and layer_name"
        )

    # Reconstruct models
    model_1 = reconstruct_model_from_task_dict(pretrained_model, task_dict_1, device=device)
    model_2 = reconstruct_model_from_task_dict(pretrained_model, task_dict_2, device=device)

    # Extract activations
    act_1 = extract_layer_activations(model_1, calibration_loader, layer_name, device)
    act_2 = extract_layer_activations(model_2, calibration_loader, layer_name, device)

    # Compute L2 distance
    distance = torch.norm(act_1 - act_2, p=2).item()

    # Clean up
    del model_1, model_2
    torch.cuda.empty_cache()

    return distance


def activation_cosine_similarity(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    pretrained_model=None,
    calibration_loader: Optional[DataLoader] = None,
    layer_name: Optional[str] = None,
    device: str = "cuda",
) -> float:
    """Compute cosine similarity between average activations of two models.

    This measures how aligned the internal representations are between
    two finetuned models in terms of direction.

    Args:
        task_dict_1: First task vector
        task_dict_2: Second task vector
        pretrained_model: Pretrained model (required for activation metrics)
        calibration_loader: DataLoader with calibration samples (required)
        layer_name: Name of layer to extract activations from (required)
        device: Device to use

    Returns:
        Cosine similarity value in [-1, 1]
    """
    if pretrained_model is None or calibration_loader is None or layer_name is None:
        raise ValueError(
            "Activation metrics require pretrained_model, calibration_loader, and layer_name"
        )

    # Reconstruct models
    model_1 = reconstruct_model_from_task_dict(pretrained_model, task_dict_1, device=device)
    model_2 = reconstruct_model_from_task_dict(pretrained_model, task_dict_2, device=device)

    # Extract activations
    act_1 = extract_layer_activations(model_1, calibration_loader, layer_name, device)
    act_2 = extract_layer_activations(model_2, calibration_loader, layer_name, device)

    # Compute cosine similarity
    similarity = F.cosine_similarity(act_1.unsqueeze(0), act_2.unsqueeze(0)).item()

    # Clean up
    del model_1, model_2
    torch.cuda.empty_cache()

    return similarity


def activation_magnitude_ratio(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    pretrained_model=None,
    calibration_loader: Optional[DataLoader] = None,
    layer_name: Optional[str] = None,
    device: str = "cuda",
) -> float:
    """Compute ratio of activation magnitudes between two models.

    This measures whether one model produces much stronger or weaker
    activations than another, which could indicate different learning scales.

    Args:
        task_dict_1: First task vector
        task_dict_2: Second task vector
        pretrained_model: Pretrained model (required for activation metrics)
        calibration_loader: DataLoader with calibration samples (required)
        layer_name: Name of layer to extract activations from (required)
        device: Device to use

    Returns:
        Magnitude ratio (smaller / larger) in (0, 1]
    """
    if pretrained_model is None or calibration_loader is None or layer_name is None:
        raise ValueError(
            "Activation metrics require pretrained_model, calibration_loader, and layer_name"
        )

    # Reconstruct models
    model_1 = reconstruct_model_from_task_dict(pretrained_model, task_dict_1, device=device)
    model_2 = reconstruct_model_from_task_dict(pretrained_model, task_dict_2, device=device)

    # Extract activations
    act_1 = extract_layer_activations(model_1, calibration_loader, layer_name, device)
    act_2 = extract_layer_activations(model_2, calibration_loader, layer_name, device)

    # Compute magnitudes
    mag_1 = torch.norm(act_1, p=2).item()
    mag_2 = torch.norm(act_2, p=2).item()

    # Compute ratio (smaller / larger)
    if mag_1 < 1e-8 or mag_2 < 1e-8:
        ratio = 0.0
    else:
        ratio = min(mag_1, mag_2) / max(mag_1, mag_2)

    # Clean up
    del model_1, model_2
    torch.cuda.empty_cache()

    return ratio


def activation_dot_product(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    pretrained_model=None,
    calibration_loader: Optional[DataLoader] = None,
    layer_name: Optional[str] = None,
    device: str = "cuda",
) -> float:
    """Compute dot product between average activations of two models.

    Unlike cosine similarity, this captures both direction and magnitude
    of the activation alignment.

    Args:
        task_dict_1: First task vector
        task_dict_2: Second task vector
        pretrained_model: Pretrained model (required for activation metrics)
        calibration_loader: DataLoader with calibration samples (required)
        layer_name: Name of layer to extract activations from (required)
        device: Device to use

    Returns:
        Dot product value
    """
    if pretrained_model is None or calibration_loader is None or layer_name is None:
        raise ValueError(
            "Activation metrics require pretrained_model, calibration_loader, and layer_name"
        )

    # Reconstruct models
    model_1 = reconstruct_model_from_task_dict(pretrained_model, task_dict_1, device=device)
    model_2 = reconstruct_model_from_task_dict(pretrained_model, task_dict_2, device=device)

    # Extract activations
    act_1 = extract_layer_activations(model_1, calibration_loader, layer_name, device)
    act_2 = extract_layer_activations(model_2, calibration_loader, layer_name, device)

    # Compute dot product
    dot_prod = torch.dot(act_1, act_2).item()

    # Clean up
    del model_1, model_2
    torch.cuda.empty_cache()

    return dot_prod


# =============================================================================
# Metric Registry
# =============================================================================


# Wrapper functions for metrics that return tuples
def right_subspace_overlap_top_k(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 10,
) -> float:
    """Wrapper for right_subspace_overlap returning only top-k overlap."""
    top_overlap, _ = right_subspace_overlap(task_dict_1, task_dict_2, top_k)
    return top_overlap


def right_subspace_overlap_bottom_k(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 10,
) -> float:
    """Wrapper for right_subspace_overlap returning only bottom-k overlap."""
    _, bottom_overlap = right_subspace_overlap(task_dict_1, task_dict_2, top_k)
    return bottom_overlap


def interaction_matrix_overlap_top_k(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 10,
) -> float:
    """Wrapper for interaction_matrix_overlap returning only top-k overlap."""
    top_overlap, _ = interaction_matrix_overlap(task_dict_1, task_dict_2, top_k)
    return top_overlap


def interaction_matrix_overlap_bottom_k(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    top_k: int = 10,
) -> float:
    """Wrapper for interaction_matrix_overlap returning only bottom-k overlap."""
    _, bottom_overlap = interaction_matrix_overlap(task_dict_1, task_dict_2, top_k)
    return bottom_overlap


# Update this registry when you add new metrics!
METRIC_REGISTRY: Dict[str, Callable] = {
    # Weight-based metrics
    "task_vector_cosine_similarity": task_vector_cosine_similarity,
    "task_vector_l2_distance": task_vector_l2_distance,
    "task_vector_dot_product": task_vector_dot_product,
    "weight_space_angle": weight_space_angle,
    "task_vector_magnitude_ratio": task_vector_magnitude_ratio,
    # Effective rank metrics (tangent space alignment)
    "effective_rank": effective_rank,
    "effective_rank_mergeability_score": effective_rank_mergeability_score,
    "stable_rank": stable_rank,
    "spectral_gap": spectral_gap,
    "singular_value_ratio": singular_value_ratio,
    "layerwise_effective_rank": layerwise_effective_rank,
    "layerwise_effective_rank_mergeability_score": layerwise_effective_rank_mergeability_score,
    # SVD-based subspace metrics
    "singular_value_overlap": singular_value_overlap,
    "subspace_overlap": subspace_overlap,
    "right_subspace_overlap": right_subspace_overlap,
    "interaction_matrix_overlap": interaction_matrix_overlap,
    "right_subspace_overlap_top_k": right_subspace_overlap_top_k,
    "right_subspace_overlap_bottom_k": right_subspace_overlap_bottom_k,
    "interaction_matrix_overlap_top_k": interaction_matrix_overlap_top_k,
    "interaction_matrix_overlap_bottom_k": interaction_matrix_overlap_bottom_k,
    # Activation-based metrics
    "activation_l2_distance": activation_l2_distance,
    "activation_cosine_similarity": activation_cosine_similarity,
    "activation_magnitude_ratio": activation_magnitude_ratio,
    "activation_dot_product": activation_dot_product,
}

# Registry for metrics that return tuples (metric_name -> list of output names)
TUPLE_METRICS: Dict[str, List[str]] = {
    "right_subspace_overlap": ["right_subspace_overlap_top_k", "right_subspace_overlap_bottom_k"],
    "interaction_matrix_overlap": ["interaction_matrix_overlap_top_k", "interaction_matrix_overlap_bottom_k"],
}


def compute_metric(
    metric_name: str,
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    **kwargs,
) -> Union[float, Dict[str, float]]:
    """Compute a specific metric by name.

    Args:
        metric_name: Name of the metric (must be in METRIC_REGISTRY).
        task_dict_1: First task vector.
        task_dict_2: Second task vector.
        **kwargs: Additional arguments passed to the metric function.

    Returns:
        Metric value (float or dict for per-layer metrics).

    Raises:
        ValueError: If metric_name is not in the registry.
    """
    if metric_name not in METRIC_REGISTRY:
        available = ", ".join(METRIC_REGISTRY.keys())
        raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {available}")

    return METRIC_REGISTRY[metric_name](task_dict_1, task_dict_2, **kwargs)


def compute_all_metrics(
    task_dict_1: Dict[str, torch.Tensor],
    task_dict_2: Dict[str, torch.Tensor],
    layer_wise: bool = False,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Compute all registered metrics.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.
        layer_wise: If True, compute all metrics per-layer and return both
                   per-layer breakdown and average.

    Returns:
        Dictionary mapping metric names to their values.
        If layer_wise=True, each metric has {"per_layer": {...}, "avg": float}
    """
    results = {}

    for name, func in METRIC_REGISTRY.items():
        try:
            if layer_wise:
                per_layer = compute_metric_per_layer(func, task_dict_1, task_dict_2)
                valid_values = [v for v in per_layer.values() if not math.isnan(v)]
                avg = sum(valid_values) / len(valid_values) if valid_values else 0.0
                results[name] = {"per_layer": per_layer, "avg": avg}
            else:
                results[name] = func(task_dict_1, task_dict_2)
        except Exception as e:
            print(f"Warning: Failed to compute {name}: {e}")
            results[name] = None

    return results
