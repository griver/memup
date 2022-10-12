from abc import ABC, abstractmethod
from torch import Tensor
import torch
from metrics.metric import Metric


class CrossEntropyMetric(Metric):

    def __init__(self):
        super().__init__("CrossEntropy")

    @torch.no_grad()
    def __call__(self, logits: Tensor, labels: Tensor) -> float:
        D = logits.shape[-1]
        return torch.nn.CrossEntropyLoss()(logits.reshape(-1, D), labels.reshape(-1)).item()