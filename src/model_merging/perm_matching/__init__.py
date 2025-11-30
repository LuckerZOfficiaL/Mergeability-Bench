"""
Permutation matching module for aligning model parameters before computing mergeability metrics.

This module provides functionality to find optimal permutations of neural network parameters
that align models before merging or computing mergeability metrics.
"""

from model_merging.perm_matching.matcher import apply_permutation_to_task_vectors
from model_merging.perm_matching.weight_matching import weight_matching_for_task_vectors

__all__ = [
    "apply_permutation_to_task_vectors",
    "weight_matching_for_task_vectors",
]
