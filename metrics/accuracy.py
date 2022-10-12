from abc import ABC, abstractmethod
from torch import Tensor
import torch
from sklearn.metrics import accuracy_score
from metrics.metric import Metric


class AccuracyMetric(Metric):

    def __init__(self):
        super().__init__("Accuracy")

    @torch.no_grad()
    def __call__(self, logits: Tensor, labels: Tensor) -> float:
        return accuracy_score(logits.argmax(-1).reshape(-1).cpu().numpy(), labels.reshape(-1).cpu().numpy())





