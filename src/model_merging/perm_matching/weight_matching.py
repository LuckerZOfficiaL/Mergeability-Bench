"""
Weight matching algorithm for finding optimal permutations between model parameters.

Based on the Git Re-Basin approach: iteratively solve linear assignment problems
to find permutations that maximize similarity between corresponding layers.
"""

import copy
import logging
from enum import auto
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from backports.strenum import StrEnum
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from tqdm import tqdm

from model_merging.permutations.permutation_spec import PermutationSpec
from model_merging.perm_matching.utils import compute_weights_similarity, get_permuted_param

pylogger = logging.getLogger(__name__)


class LayerIterationOrder(StrEnum):
    RANDOM = auto()
    FORWARD = auto()
    BACKWARD = auto()


def solve_linear_assignment_problem(sim_matrix: Union[torch.Tensor, np.ndarray]) -> Tensor:
    """Solve the linear assignment problem to find optimal permutation."""
    if isinstance(sim_matrix, torch.Tensor):
        sim_matrix = sim_matrix.cpu().detach().numpy()

    ri, ci = linear_sum_assignment(sim_matrix, maximize=True)
    assert (torch.tensor(ri) == torch.arange(len(ri))).all()

    indices = torch.tensor(ci)
    return indices


def get_layer_iteration_order(layer_iteration_order: LayerIterationOrder, num_layers: int):
    """Get the order in which to iterate through layers."""
    if layer_iteration_order == LayerIterationOrder.RANDOM:
        return torch.randperm(num_layers)
    elif layer_iteration_order == LayerIterationOrder.FORWARD:
        return torch.arange(num_layers)
    elif layer_iteration_order == LayerIterationOrder.BACKWARD:
        return range(num_layers)[num_layers:0:-1]
    else:
        raise NotImplementedError(f"Unknown layer iteration order {layer_iteration_order}")


def weight_matching_for_task_vectors(
    ps: PermutationSpec,
    pretrained_params: Dict[str, Tensor],
    fixed_finetuned_params: Dict[str, Tensor],
    permutee_finetuned_params: Dict[str, Tensor],
    max_iter: int = 100,
    init_perm=None,
    layer_iteration_order: LayerIterationOrder = LayerIterationOrder.RANDOM,
    verbose: bool = False,
) -> Dict[str, Tensor]:
    """
    Find a permutation of permutee parameters to align with fixed parameters.

    This function works with task vectors (differences from pretrained model).

    Args:
        ps: PermutationSpec defining the permutation structure
        pretrained_params: Pretrained model parameters (used to compute task vectors)
        fixed_finetuned_params: Fine-tuned parameters to match against (fixed)
        permutee_finetuned_params: Fine-tuned parameters to permute
        max_iter: Maximum number of iterations
        init_perm: Initial permutation (None for identity)
        layer_iteration_order: Order to iterate through layers
        verbose: Whether to print detailed progress

    Returns:
        Dictionary of permutation indices for each layer
    """
    if not verbose:
        pylogger.setLevel(logging.WARNING)

    # Compute task vectors (differences from pretrained)
    fixed_task_vector = {
        k: fixed_finetuned_params[k] - pretrained_params[k]
        for k in fixed_finetuned_params.keys()
        if k in pretrained_params
    }

    permutee_task_vector = {
        k: permutee_finetuned_params[k] - pretrained_params[k]
        for k in permutee_finetuned_params.keys()
        if k in pretrained_params
    }

    # Determine permutation sizes from the fixed model
    perm_sizes = {}
    for p, params_and_axes in ps.perm_to_layers_and_axes.items():
        # Get reference parameter and axis
        ref_tuple = params_and_axes[0]
        ref_param_name = ref_tuple[0]
        ref_axis = ref_tuple[1]

        if ref_param_name in fixed_task_vector:
            perm_sizes[p] = fixed_task_vector[ref_param_name].shape[ref_axis]

    # Initialize with identity permutation if none given
    all_perm_indices = (
        {p: torch.arange(n) for p, n in perm_sizes.items()}
        if init_perm is None
        else init_perm
    )

    perm_names = list(all_perm_indices.keys())
    num_layers = len(perm_names)

    pylogger.info(f"Starting weight matching with {num_layers} permutation layers")
    pylogger.info(f"Permutation sizes: {perm_sizes}")

    # Iteratively refine permutations
    for iteration in tqdm(range(max_iter), desc="Weight matching", disable=not verbose):
        progress = False
        perm_order = get_layer_iteration_order(layer_iteration_order, num_layers)

        for p_ix in perm_order:
            p = perm_names[p_ix]
            num_neurons = perm_sizes[p]

            # Initialize similarity matrix
            sim_matrix = torch.zeros((num_neurons, num_neurons))

            # Get all parameters affected by this permutation
            params_and_axes: List[Tuple[str, int]] = ps.perm_to_layers_and_axes[p]

            for params_name, axis in params_and_axes:
                # Skip if parameter not in task vectors
                if params_name not in fixed_task_vector or params_name not in permutee_task_vector:
                    continue

                w_a = copy.deepcopy(fixed_task_vector[params_name])
                w_b = copy.deepcopy(permutee_task_vector[params_name])

                if w_a.shape != w_b.shape:
                    pylogger.warning(f"Shape mismatch for {params_name}: {w_a.shape} vs {w_b.shape}")
                    continue

                # Get permutations to apply
                perms_to_apply = ps.layer_and_axes_to_perm[params_name]

                # Apply all permutations except the current axis
                w_b = get_permuted_param(
                    w_b, perms_to_apply, all_perm_indices, except_axis=axis
                )

                # Reshape to (num_neurons, -1) for similarity computation
                w_a = torch.moveaxis(w_a, axis, 0).reshape((num_neurons, -1))
                w_b = torch.moveaxis(w_b, axis, 0).reshape((num_neurons, -1))

                # Accumulate similarity: maximize w_a @ w_b.T
                sim_matrix += w_a @ w_b.T

            # Solve linear assignment problem to find best permutation
            perm_indices = solve_linear_assignment_problem(sim_matrix)

            # Check if this improves the solution
            old_similarity = compute_weights_similarity(sim_matrix, all_perm_indices[p])
            new_similarity = compute_weights_similarity(sim_matrix, perm_indices)

            similarity_improvement = new_similarity - old_similarity
            pylogger.info(f"Iteration {iteration}, Permutation {p}: improvement = {similarity_improvement:.6f}")

            # Update permutation if it improves
            all_perm_indices[p] = perm_indices

            progress = progress or similarity_improvement > 1e-12

        # Stop if no progress
        if not progress:
            pylogger.info(f"Converged after {iteration + 1} iterations")
            break

    return all_perm_indices
