from abc import ABC, abstractmethod
import numpy as np
from datagen import TrajectoryPreprocessor, Trajectory
from .batch import MemupBatch
from typing import List, Optional
import torch


class UncertaintyDetector(TrajectoryPreprocessor):
    """
    Processes a trajectory and add new uncertainty_key that contains
    uncertainty estimates.
    """
    def __init__(self, uncertainty_key='uncertainty'):
        self.uncertainty_key=uncertainty_key

    @abstractmethod
    def make_estimate(self, trajectories: List[Trajectory]) -> List[List[float]]:
        pass

    def process_single_(self, trajectory: Trajectory) -> Trajectory:
        estimate, = self.make_estimate([trajectory])
        trajectory.data[self.uncertainty_key] = estimate
        return trajectory

    def process_(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        estimates = self.make_estimate(trajectories)

        for est, tr in zip(estimates, trajectories):
            tr.data[self.uncertainty_key]=est

        return trajectories


class DummyTailDetector(UncertaintyDetector):
    """
    DummyTailDetector allows to assign high uncertainty to the last <tail_length> elements of a sequence.
    It is used for testing and debug in tasks like
    t-maze and copy.
    """
    TAIL_WEIGHT = 20

    def __init__(self, tail_length=10, uncertainty_key='uncertainty', target_key: Optional[str] = None):
        super().__init__(uncertainty_key)
        self.tail_length = tail_length
        self.target_key = target_key

    def make_estimate(self, trajectories):
        uncertainty = []
        for e in trajectories:
            length = len(e) if self.target_key is None else len(e.data[self.target_key])
            unc = np.zeros(length, dtype=np.float32)
            unc[-self.tail_length:] += self.TAIL_WEIGHT
            uncertainty.append(unc)

        return uncertainty


class PredictionErrorBasedDetector(UncertaintyDetector):
    """
    This detector uses surprise as a single point uncertainty estimate:
    -log p(y). For CE loss and MSE loss this simply means computing model's
    error on a given sample.
    Writes surprise value into trajectory.data by uncertainty_key
    If context_key is provided, also adds memory's hidden states
    into trajectory.data
    """
    def __init__(
            self, memup_memory, memup_predictor, error_metric,
            context_key=None,
            uncertainty_key='uncertainty'  # rollout_len=100
    ):
        super().__init__(uncertainty_key)
        self.memup_memory = memup_memory
        self.memup_predictor = memup_predictor
        self.error_metric = error_metric
        self.context_key = context_key
        # self.rollout_len = rollout_len

    @torch.no_grad()
    def make_estimate(self, trajectories):
        lengths = [len(t) for t in trajectories]
        idx = [np.arange(l) for l in lengths]

        full_batch = MemupBatch(
            trajectories,
            mem_idx=idx,
            pred_idx=idx,
            target_idx=[e.reshape(-1, 1) for e in idx]
        )
        mem_state = self.memup_memory.init_state(len(trajectories))
        mem_output, _ = self.memup_memory.process_batch(full_batch, mem_state)
        if self.context_key:
            context = torch.split(mem_output.cpu(), lengths)
            for c, t in zip(context, trajectories):
                t.data[self.context_key] = c

        pred_state = self.memup_predictor.init_state(len(full_batch.trajs))

        out = self.memup_predictor.process_batch(full_batch, mem_output, pred_state)

        errors = self.error_metric(
            out['preds'],#.flatten(0, 1),
            out['targets']#.flatten(0, 1)
        ).reshape(-1)

        errors = torch.split(errors.cpu(), lengths)

        return errors

    def process(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        estimates = self.make_estimate(trajectories)

        for i, tr in enumerate(trajectories):
            tr.data[self.uncertainty_key] = estimates[i]
            # if self.context_key:
            #    tr.data[self.context_key] = contexts[i]

        return trajectories

    def process_single(self, trajectory: Trajectory) -> Trajectory:
        estimates = self.make_estimate([trajectory])
        trajectory.data[self.uncertainty_key] = estimates[0]
        # if self.context_key:
        #    trajectory.data[self.context_key] = contexts[0]

        return trajectory

