import copy
import logging
from collections import OrderedDict
from typing import Dict

import torch
from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
)

pylogger = logging.getLogger(__name__)


def _topk_values_mask(tensor: torch.Tensor, k: float) -> torch.Tensor:
    """
    Keep only the top-k% values by magnitude, zero out the rest.

    Args:
        tensor: Input tensor (can be 1D or 2D)
        k: Fraction of values to keep (0 to 1), e.g., 0.2 keeps top 20%
    """
    if k >= 1.0:
        return tensor

    original_shape = tensor.shape
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)

    n, d = tensor.shape
    num_to_keep = int(d * k)
    num_to_keep = max(num_to_keep, 1)  # Keep at least 1 element

    # Find the threshold value (k-th largest by magnitude)
    kth_values, _ = tensor.abs().kthvalue(d - num_to_keep + 1, dim=1, keepdim=True)

    # Create mask for values >= threshold
    mask = tensor.abs() >= kth_values
    result = tensor * mask

    if original_shape != result.shape:
        result = result.squeeze(0)

    return result


def _resolve_sign(tensor: torch.Tensor, method: str = "mass") -> torch.Tensor:
    """
    Resolve sign conflicts across task vectors.

    Args:
        tensor: 2D tensor of shape (num_tasks, num_params)
        method: Sign resolution method ("mass", "normfrac", or "normmass")

    Returns:
        1D tensor of resolved signs for each parameter
    """
    if method == "mass":
        # Simple majority vote by sum
        sign_to_mult = torch.sign(tensor.sum(dim=0))
    elif method == "normfrac":
        # Sign from the task with highest contribution to norm for each param
        row_norms = torch.norm(tensor, dim=1, keepdim=True)
        norm_fracs = (tensor ** 2) / (row_norms ** 2 + 1e-10)
        sign_to_mult = torch.sign(tensor[norm_fracs.argmax(dim=0), torch.arange(tensor.shape[1])])
    elif method == "normmass":
        # Weighted by both sign and norm mass
        row_norms = torch.norm(tensor, dim=1, keepdim=True)
        norm_fracs = (tensor ** 2) / (row_norms ** 2 + 1e-10)
        sign_to_mult = (tensor.sign() * norm_fracs.abs()).sum(dim=0).sign()
    else:
        raise ValueError(f"Unknown sign resolution method: {method}")

    # Resolve zeros to majority sign
    majority_sign = torch.sign(sign_to_mult.sum())
    if majority_sign == 0:
        majority_sign = 1.0
    sign_to_mult[sign_to_mult == 0] = majority_sign

    return sign_to_mult


def _disjoint_merge(tensor: torch.Tensor, sign_to_mult: torch.Tensor, merge_func: str = "mean") -> torch.Tensor:
    """
    Merge task vectors by only aggregating parameters with agreed signs.

    Args:
        tensor: 2D tensor of shape (num_tasks, num_params)
        sign_to_mult: 1D tensor of resolved signs
        merge_func: Aggregation method ("mean" or "sum")

    Returns:
        1D tensor of merged parameters
    """
    # Select entries that match the resolved sign
    rows_to_keep = torch.where(
        sign_to_mult.unsqueeze(0) > 0,
        tensor > 0,
        tensor < 0
    )
    selected_entries = tensor * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        merged = torch.sum(selected_entries, dim=0) / torch.clamp(non_zero_counts, min=1)
    elif merge_func == "sum":
        merged = torch.sum(selected_entries, dim=0)
    else:
        raise ValueError(f"Unknown merge function: {merge_func}")

    return merged


class TIESMerger(TaskVectorBasedMerger):
    """
    TIES Merging: Resolving Interference When Merging Models (NeurIPS 2023)

    TIES performs three steps:
    1. Trim: Remove low-magnitude parameters (keep top-k%)
    2. Elect: Resolve sign conflicts via majority vote
    3. Merge: Aggregate only parameters with agreed signs (disjoint merge)
    """

    def __init__(
        self,
        scaling_coefficient: float = 0.3,
        k: float = 0.2,
        resolve_method: str = "mass",
        merge_func: str = "mean",
        device: str = "cuda",
    ):
        """
        Args:
            scaling_coefficient: Scaling factor for the merged task vector (like alpha in Task Arithmetic)
            k: Fraction of parameters to keep after trimming (0 to 1), default 0.2 (top 20%)
            resolve_method: Sign resolution method ("mass", "normfrac", "normmass")
            merge_func: Aggregation method ("mean" or "sum")
            device: Device to use for computation
        """
        super().__init__()

        self.scaling_coefficient = scaling_coefficient
        self.k = k
        self.resolve_method = resolve_method
        self.merge_func = merge_func
        self.device = device

    def merge(
        self, base_model: ImageEncoder, finetuned_models: Dict[str, Dict]
    ) -> ImageEncoder:

        datasets = list(finetuned_models.keys())
        pretrained_model = copy.deepcopy(base_model)
        base_state_dict = base_model.state_dict()

        # Compute task vectors for each finetuned model
        task_vectors = []
        for dataset in datasets:
            task_dict = compute_task_dict(base_state_dict, finetuned_models[dataset])
            task_vectors.append(task_dict)
            del finetuned_models[dataset]
            torch.cuda.empty_cache()

        # Get all parameter keys
        param_keys = list(task_vectors[0].keys())

        # Merged task vector dictionary
        merged_task_vector = OrderedDict()

        # Process each parameter
        for key in param_keys:
            # Stack task vectors for this parameter: shape (num_tasks, param_size)
            stacked = torch.stack([tv[key].flatten().float() for tv in task_vectors], dim=0)

            # Step 1: Trim - keep only top-k% by magnitude
            trimmed = torch.stack([_topk_values_mask(row, self.k) for row in stacked], dim=0)

            # Step 2: Elect - resolve sign conflicts
            final_signs = _resolve_sign(trimmed, self.resolve_method)

            # Step 3: Merge - disjoint aggregation
            merged = _disjoint_merge(trimmed, final_signs, self.merge_func)

            # Reshape back to original shape
            original_shape = task_vectors[0][key].shape
            merged_task_vector[key] = merged.view(original_shape).to(task_vectors[0][key].dtype)

        # Apply merged task vector to pretrained model
        merged_encoder = apply_dict_to_model(
            merged_task_vector, pretrained_model, coefficient=self.scaling_coefficient
        )

        return merged_encoder
