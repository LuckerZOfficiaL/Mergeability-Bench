"""
Mergeability metrics for predicting model merging outcomes.

This module provides various metrics to measure the compatibility/similarity
between two task vectors, which can be used to predict the success of model merging.
"""

import math
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


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
) -> float:
    """Compute principal right subspace overlap between task vectors.

    Measures how much the principal right directions (from SVD, using V matrices)
    of the two task vectors overlap. High overlap might indicate task compatibility.

    Args:
        task_dict_1: First task vector.
        task_dict_2: Second task vector.
        top_k: Number of top principal directions to consider.

    Returns:
        Average right subspace overlap across all 2D weight matrices.
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
            _, _, v1 = torch.linalg.svd(tensor1.float(), full_matrices=False)
            _, _, v2 = torch.linalg.svd(tensor2.float(), full_matrices=False)
        except Exception:
            continue

        # Take top-k rows of V matrices (V is returned as V^H in torch.linalg.svd)
        k = min(top_k, v1.shape[0], v2.shape[0])
        v1_k = v1[:k, :]
        v2_k = v2[:k, :]

        # Compute subspace overlap using Frobenius norm of V1 @ V2^T
        # Maximum overlap is sqrt(k) when subspaces are identical
        product = v1_k @ v2_k.T
        overlap = torch.norm(product, p='fro').item() / k
        overlaps.append(overlap)

    if not overlaps:
        return 0.0

    return sum(overlaps) / len(overlaps)


# =============================================================================
# Metric Registry
# =============================================================================


# Update this registry when you add new metrics!
METRIC_REGISTRY: Dict[str, Callable] = {
    "task_vector_cosine_similarity": task_vector_cosine_similarity,
    "task_vector_l2_distance": task_vector_l2_distance,
    "task_vector_dot_product": task_vector_dot_product,
    "weight_space_angle": weight_space_angle,
    "task_vector_magnitude_ratio": task_vector_magnitude_ratio,
    "singular_value_overlap": singular_value_overlap,
    "subspace_overlap": subspace_overlap,
    "right_subspace_overlap": right_subspace_overlap,
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
