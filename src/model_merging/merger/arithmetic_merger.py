import copy
import logging
from typing import Dict, List
import torch
from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    sum_task_dict,
)

pylogger = logging.getLogger(__name__)


class TaskArithmeticMerger(TaskVectorBasedMerger):

    def __init__(self, optimal_alpha, device="cuda"):
        super().__init__()

        self.optimal_alpha = optimal_alpha
        self.device = device

    def merge(
        self, base_model: ImageEncoder, finetuned_models: Dict[str, Dict]
    ) -> ImageEncoder:

        cumulative_dict = {}

        datasets = list(finetuned_models.keys())
        pretrained_model = copy.deepcopy(base_model)

        for dataset in datasets:
            cumulative_dict = sum_task_dict(
                cumulative_dict,
                compute_task_dict(
                    base_model.state_dict(), finetuned_models[dataset]
                ),
            )
            del finetuned_models[dataset]  # Delete one model at a time
            torch.cuda.empty_cache()

        merged_encoder = apply_dict_to_model(
            cumulative_dict, pretrained_model, coefficient=self.optimal_alpha
        )

        return merged_encoder
