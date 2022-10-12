from abc import ABC, abstractmethod
from itertools import chain

import numpy as np

from . import MemUPMemory
from .batch import TrajectorySlicer, MemupBatch, Trajectory
from datagen import TrajectoryPreprocessor
from .batch import TrajectorySlicer
from .nets import PredictorModule, PredictorModuleWithContext
import torch.nn as nn
from typing import Tuple, Any, List, Sequence
import torch
from torch import Tensor


class AbstractMemUPPredictor(ABC):

    def __init__(self, input_keys: Sequence[str], target_key: str = 'y'):
        self.target_key = target_key
        self.input_keys = input_keys
        self.preprocessor = TrajectorySlicer(
            [*self.input_keys, self.target_key]
        )

    @abstractmethod
    def device(self): pass

    @staticmethod
    def expand_memory(memory_output: Tensor, target_idx: List[Any]):
        index = torch.cat([
            torch.ones(len(idx), device=memory_output.device, dtype=torch.int64) * i
            for i, idx in enumerate(chain(*target_idx))
        ])
        return memory_output[index]

    def extract_batch_data(self, batch: MemupBatch):
        flat_target_index = [list(chain(*idx)) for idx in batch.target_idx]
        target_events = self.preprocessor.process(batch.trajs, flat_target_index, self.device)

        targets = target_events.pop(self.target_key, None)
        targets = torch.cat(targets)

        return target_events, targets

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def train(self):
        pass


class MemUPPredictor(AbstractMemUPPredictor):

    def __init__(self, predictor_module: PredictorModule, target_key: str = 'y'):

        self.predictor = predictor_module
        super().__init__(self.predictor.input_keys(), target_key)

    @property
    def device(self):
        return self.predictor.device


    def process_batch(self, batch, memory_output, *args, **kwargs):
        """
        Processes MemupBatch and memory states to make predictions.
        Returns predictions and targets
        """
        memory_expanded = RNNMemUPPredictor.expand_memory(memory_output, batch.target_idx)
        pred_inputs, pred_targets = self.extract_batch_data(batch)

        Nm = memory_expanded.size(0)
        Nt = pred_targets.shape[0]
        assert Nm == Nt, 'context and memory should have the same number of entries'

        preds = self.predictor(
            pred_inputs, memory_expanded
        )

        return dict(preds=preds, targets=pred_targets)

    def eval(self):
        return self.predictor.eval()

    def train(self):
        return self.predictor.train()

    def init_state(self, *args, **kwargs):
        return None

    def detach_state(self, *args, **kwargs):
        return None


class RNNMemUPPredictor(AbstractMemUPPredictor):

    def __init__(self, predictor_module: PredictorModuleWithContext, target_key: str = 'y'):
        super().__init__(predictor_module.input_keys(), target_key)
        self.predictor = predictor_module

    @property
    def device(self):
        return self.predictor.device

    def filter_output_for_prediction(
            self,
            output,
            batch
    ):
        res_ids = []
        L = output.shape[1]
        shift = 0
        for idx, m_idx in zip(batch.pred_idx, batch.mem_idx):
            th_idx = torch.as_tensor(idx) - m_idx[0] + shift
            res_ids.append(th_idx)
            shift += L

        res_ids = torch.cat(res_ids).type(torch.long).to(self.device)

        return output.flatten(0, 1)[res_ids]

    def predict_context(self, batch, state):
        preprocessor = TrajectorySlicer(
            ['x', 'done']
        )

        target_events = preprocessor.process(batch.trajs, batch.target_idx, self.device)

        context, new_state = self.predictor.predict_context(
            target_events, state
        )

        return self.filter_output_for_prediction(context, batch), new_state

    def process_batch(self, batch: MemupBatch, memory_output: Tensor, pred_state: Tuple[Tensor, Tensor]):

        """
        Processes MemupBatch and memory states to make predictions.
        Returns predictions and targets
        """
        memory_output_expanded = RNNMemUPPredictor.expand_memory(memory_output, batch.target_idx)
        input_data, targets = self.extract_batch_data(batch)

        Nm, D = memory_output_expanded.size()
        Nt = targets.shape[0]
        assert Nm == Nt, 'context and memory should have the same number of entries'

        preds, context, pred_state = self.predictor(
            input_data, memory_output_expanded, pred_state
        )

        return dict(
            preds=preds,
            context=context,
            pred_state=pred_state,
            targets=targets
        )

    def init_state(self, *args, **kwargs):
        return self.predictor.init_state(*args, **kwargs)

    def detach_state(self, *args, **kwargs):
        return self.predictor.detach_state(*args, **kwargs)

    def eval(self):
        return self.predictor.eval()

    def train(self):
        return self.predictor.train()


class AccumulatedContextPreprocessor(TrajectoryPreprocessor):

    def __init__(self, memup_predictor: RNNMemUPPredictor, key='context'):
        self.memup_predictor = memup_predictor
        self.key = key

    @torch.no_grad()
    def process_(self, trajectories):
        lengths = [len(t) for t in trajectories]
        idx = [np.arange(l) for l in lengths]

        full_batch = MemupBatch(
            trajectories,
            mem_idx=idx,
            pred_idx=idx,
            target_idx=[e.reshape(-1, 1) for e in idx]
        )
        state = self.memup_predictor.init_state(len(trajectories))
        context, new_state = self.memup_predictor.predict_context(full_batch, state)
        context = torch.split(context.cpu(), lengths)
        for c, t in zip(context, trajectories):
            t.data[self.key] = c

        return trajectories

    def process_single_(self, trajectory: Trajectory) -> Trajectory:
        raise NotImplementedError()
