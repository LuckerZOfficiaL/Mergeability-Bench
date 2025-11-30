"""
Utility functions for permutation matching.
"""

import copy
import torch
from torch import Tensor
from typing import Dict

# Type aliases
PermutationMatrix = Tensor  # shape (n, n)
PermutationIndices = Tensor  # shape (n,)


def compute_weights_similarity(similarity_matrix: Tensor, perm_indices: PermutationIndices) -> Tensor:
    """Compute the total similarity given a similarity matrix and permutation.

    Args:
        similarity_matrix: Matrix where S[i, j] represents similarity between neurons i and j
        perm_indices: Permutation indices

    Returns:
        Total similarity score
    """
    n = len(perm_indices)
    similarity = torch.sum(similarity_matrix[torch.arange(n), perm_indices.long()])
    return similarity


def get_permuted_param(param: Tensor, perms_to_apply: tuple, perm_matrices: Dict[str, PermutationIndices], except_axis=None) -> Tensor:
    """Apply permutations to a parameter tensor.

    Args:
        param: The parameter tensor to permute
        perms_to_apply: Tuple of permutation identifiers for each axis
        perm_matrices: Dictionary mapping permutation IDs to permutation indices
        except_axis: Axis to skip when applying permutations

    Returns:
        Permuted parameter tensor
    """
    for axis, perm_id in enumerate(perms_to_apply):
        if axis == except_axis or perm_id is None:
            continue

        perm = perm_matrices[perm_id].cpu()
        param = param.cpu()

        # Permute by selecting indices along the specified axis
        param = torch.index_select(param, axis, perm.long())

    return param


def apply_permutation_to_statedict(ps, perm_matrices: Dict[str, PermutationIndices], all_params: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Apply permutations to all parameters in a state dict.

    Args:
        ps: PermutationSpec defining the permutation structure
        perm_matrices: Dictionary of permutation indices for each layer
        all_params: State dictionary to permute

    Returns:
        Permuted state dictionary
    """
    permuted_params = {}

    for param_name, param in all_params.items():
        param_name_in_perm_dict = param_name

        # Skip special parameters that shouldn't be permuted
        if "num_batches_tracked" in param_name or "temperature" in param_name or "to_patch_tokens.1" in param_name:
            permuted_params[param_name] = param
            continue

        # Handle running mean/var specially
        if "running_mean" in param_name or "running_var" in param_name:
            layer_name = ".".join(param_name.split(".")[:-1])
            param_name_in_perm_dict = layer_name + ".weight"

        # If parameter is not in the permutation spec, keep it as is
        if param_name_in_perm_dict not in ps.layer_and_axes_to_perm:
            permuted_params[param_name] = param
            continue

        param = copy.deepcopy(param)
        perms_to_apply = ps.layer_and_axes_to_perm[param_name_in_perm_dict]

        param = get_permuted_param(param, perms_to_apply, perm_matrices)
        permuted_params[param_name] = param

    return permuted_params
