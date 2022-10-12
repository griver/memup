from abc import ABC, abstractmethod
from torch import Tensor
import torch
from metrics.metric import Metric


class MSEMetric(Metric):

    def __init__(self):
        super().__init__("MSE")

    @torch.no_grad()
    def __call__(self, preds: Tensor, labels: Tensor) -> float:
        return torch.nn.MSELoss()(preds.reshape(-1), labels.reshape(-1)).item()