from itertools import chain
from typing import Any, Dict, List
import torch
import numpy as np
import random as rnd
import os
from itertools import chain
from typing import Callable, Tuple, Iterator, List, Any
from torch import nn, Tensor
from torch.nn import Parameter
import torch
from datagen import TrajectoryBuffer, RandomSampler, OrderedSampler
from datagen.preprocessing import Composite
from memup import nets, MemUPMemory, MemUPPredictor, MemupBatch, PredictionErrorBasedDetector, TruncatedMemUPSampler, \
    TopKSelector, CompositeSelector, CurrentStepSelector, MemUPEvalSampler
from memup.memory import AccumulatedMemoryPreprocessor
from memup.nets import accumulate
from metrics import Metric
from .predictor import RNNMemUPPredictor

"""
The idea was to create a class for memup training 
but just for now i use only these functions:
"""

def update_on_batch(
        batch: MemupBatch,
        memup_memory: MemUPMemory,
        memup_predictor: MemUPPredictor,
        mem_state: Any,
        criterion: Any,
        optimizer: torch.optim.Optimizer,
        clip_grad: float = 1.,
        multiplier: float = 1,
        min_loss_threshold=0.0  #0.00002
    ):
    """
    Processes a MemUPBatch and makes a single step of optimization.
    Returns a 2-tuple where the first element is the value of
    loss function at this batch and the second element
    contains a memory state at the end of each processed subsequence.
    """
    mem_output, mem_state = memup_memory.process_batch(batch, mem_state)
    output = memup_predictor.process_batch(batch, mem_output)
    preds = output['preds'] #.flatten(0, 1)
    targets = output['targets'] #.flatten(0,1)

    loss = criterion(preds, targets)
    if loss.item() >= min_loss_threshold:

        optimizer.zero_grad()
        (loss * multiplier).backward()

        if clip_grad:
            params = chain(
                memup_memory.memory.parameters(),
                memup_predictor.predictor.parameters()
            )
            torch.nn.utils.clip_grad_norm_(params, clip_grad)

        optimizer.step()

    mem_state = memup_memory.detach_state(mem_state)

    return dict(
        loss=loss.item(),
        mem_state=mem_state,
        preds=preds.detach().cpu().numpy(),
        targets=targets.detach().cpu().numpy()
    )


def update_on_batch_two_rnn(
        batch: MemupBatch,
        memup_memory: MemUPMemory,
        memup_predictor: RNNMemUPPredictor,
        mem_state: Any,
        pred_state: Any,
        criterion: Any,
        optimizer: torch.optim.Optimizer,
        clip_grad: float = 1.,
        multiplier: float = 1
    ):
    """
    Processes a MemUPBatch and makes a single step of optimization.
    Returns a 2-tuple where the first element is the value of
    loss function at this batch and the second element
    contains a memory state at the end of each processed subsequence.
    """
    mem_output, mem_state = memup_memory.process_batch(batch, mem_state)
    pred_out = memup_predictor.process_batch(batch, mem_output, pred_state)
    pred_state =pred_out['pred_state']

    loss = criterion(pred_out['preds'], pred_out['targets'])

    optimizer.zero_grad()
    (loss * multiplier).backward()

    if clip_grad:
        params = chain(
            memup_memory.memory.parameters(),
            memup_predictor.predictor.parameters()
        )
        torch.nn.utils.clip_grad_norm_(params, clip_grad)

    optimizer.step()

    mem_state = memup_memory.detach_state(mem_state)
    if pred_state:
        pred_state = memup_memory.detach_state(pred_state)

    return dict(loss=loss.item(), mem_state=mem_state, pred_state=pred_state)


@torch.no_grad()
def eval_memory_and_predictor(
        eval_sampler: MemUPEvalSampler,
        memup_memory: MemUPMemory,
        memup_predictor: MemUPPredictor,
        metrics: List[Metric],
        batch_size: int=100,
) -> dict:
    """
    Returns a metric value on trajectories in eval_sampler.
    For more details see: datagen.MemUPEvalSampler
    """
    memup_memory.eval()
    memup_predictor.eval()

    all_preds = []
    all_targets = []

    for full_batch in eval_sampler.epoch(batch_size):
        mem_state = memup_memory.init_state(len(full_batch.trajs))

        mem_out, _ = memup_memory.process_batch(full_batch, mem_state)
        out = memup_predictor.process_batch(full_batch, mem_out)
        all_preds.append(out['preds'])
        all_targets.append(out['targets'])

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    memup_memory.train()
    memup_predictor.train()
    result = {m.name:m(all_preds, all_targets) for m in metrics}
    result['preds'] = all_preds.detach().cpu().numpy()
    result['targets'] = all_targets.detach().cpu().numpy()

    return result


@torch.no_grad()
def eval_memory_and_predictor_two_rnn(
        eval_sampler: MemUPEvalSampler,
        memup_memory: MemUPMemory,
        memup_predictor: RNNMemUPPredictor,
        metrics: List[Metric],
        batch_size: int=100,
) -> Dict[str, float]:
    """
    Returns a metric value on trajectories in eval_sampler.
    For more details see: datagen.MemUPEvalSampler
    """
    memup_memory.eval()
    memup_predictor.eval()
    all_preds = []
    all_targets = []

    for full_batch in eval_sampler.epoch(batch_size):

        mem_state = memup_memory.init_state(len(full_batch.trajs))
        output, _ = memup_memory.process_batch(full_batch, mem_state)
        pred_state = memup_predictor.init_state(len(full_batch.trajs))
        out = memup_predictor.process_batch(full_batch, output, pred_state)
        all_preds.append(out['preds'])
        all_targets.append(out['targets'])

    memup_memory.train()
    memup_predictor.train()
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return {m.name: m(all_preds, all_targets) for m in metrics}


def fix_seed(seed, env=None, test_env=None):
    """
    Fix seed for reproducibility, but... as said in PyTorch docs:

    Completely reproducible results are not guaranteed across PyTorch
    releases, individual commits, or different platforms. Furthermore,
    results may not be reproducible between CPU and GPU executions,
    even when using identical seeds.
    """
    msg = "'SEED({}) is fixed for pytorch, numpy, and std lib".format(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rnd.seed(seed)

    if env:
        msg += ', env'
        env.seed(seed)
    if test_env:
        msg += " and test_env"
        test_env.seed(2 ** 31 - 1 - seed)

    print(msg)


def ensure_dir(file_path):
    """
    Checks if the containing directories exist,
    and if not, creates them.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


class MemupModule:

    def __init__(self,
                 create_memory: Callable[[], nets.RecurrentModule],
                 create_predictor: Callable[[], nets.PredictorModule],
                 target_key: str,
                 detector_metric: nn.Module):
        self.memory_net = create_memory()
        self.memup_memory = MemUPMemory(self.memory_net)

        self.predictor_net = create_predictor()
        self.memup_predictor = MemUPPredictor(self.predictor_net, target_key)

        self.predictor_acc = create_predictor()
        self.memory_acc = create_memory()

        self.detector = PredictionErrorBasedDetector(
            MemUPMemory(self.memory_acc),
            MemUPPredictor(self.predictor_acc, target_key),
            error_metric=detector_metric,
            context_key=None
        )

    def parameters(self) -> Iterator[Parameter]:
        return chain(self.memory_net.parameters(), self.predictor_net.parameters())

    def eval(self):
        self.memup_memory.eval()
        self.memup_predictor.eval()

    def train(self):
        self.memup_memory.train()
        self.memup_predictor.train()

    def init_state(self, batch_size: int):
        return self.memup_memory.init_state(batch_size)

    def accumulate(self):
        accumulate(self.memory_net, self.memory_acc, 0.995)
        accumulate(self.predictor_net, self.predictor_acc, 0.995)

    def process_batch(self, batch: MemupBatch, mem_state: Tuple[Tensor, ...]) -> Tuple[Tensor, Tensor, Tuple[Tensor, ...]]:
        mem_output, mem_state = self.memup_memory.process_batch(batch, mem_state)
        output = self.memup_predictor.process_batch(batch, mem_output)
        preds = output['preds']
        targets = output['targets']
        mem_state = self.memup_memory.detach_state(mem_state)

        return preds, targets, mem_state


class MemupTrainOp:

    def __init__(self, memup_module: MemupModule, buffer: TrajectoryBuffer, pred_freq: int, rollout: int, criterion: nn.Module):
        self.memup_module = memup_module
        self.criterion = criterion

        self.sampler = TruncatedMemUPSampler(
            RandomSampler(buffer),
            CompositeSelector([
                TopKSelector(1, time_dependent_selection=True),
                CurrentStepSelector()
            ]),
            Composite([
                AccumulatedMemoryPreprocessor(MemUPMemory(memup_module.memory_acc), "context"),
                memup_module.detector
            ]),
            prediction_frequency=pred_freq,
            rollout=rollout,
        )

        self.optimizer = torch.optim.Adam(memup_module.parameters(), lr=5e-4)
        self.state = None
        self.postproc = []

    def foreach(self, f: Callable[[MemupBatch], Any]):
        self.postproc.append(f)

    def exec(self, batch_size: int, num_batches: int, clip_grad: float = 1.0):

        if self.state is None:
            self.state = self.memup_module.init_state(batch_size)

        mem_state = self.state
        total_loss = 0
        epoch_batches = 0

        for batch in self.sampler.epoch(batch_size, num_batches):
            epoch_batches += 1
            self.optimizer.zero_grad()
            preds, targets, mem_state = self.memup_module.process_batch(batch, mem_state)
            loss = self.criterion(preds, targets)
            loss.backward()

            if clip_grad:
                params = self.memup_module.parameters()
                torch.nn.utils.clip_grad_norm_(params, clip_grad)

            self.optimizer.step()

            total_loss += loss.item()
            self.memup_module.accumulate()

            print('\rloss: {:.4f}, num_batches: {}'.format(
                total_loss / epoch_batches, epoch_batches), end=''
            )

            for proc in self.postproc:
                proc(batch)

        self.state = mem_state
        return total_loss


class MemupEvalOp:

    def __init__(self, memup_module: MemupModule, buffer: TrajectoryBuffer, metrics: List[Metric]):

        self.memup_module = memup_module
        self.metrics = metrics
        self.sampler = MemUPEvalSampler(
            OrderedSampler(buffer),
            dynamic_preprocessor=memup_module.detector
        )

    def exec(self, batch_size: int):
        self.memup_module.eval()

        all_preds = []
        all_targets = []

        for full_batch in self.sampler.epoch(batch_size):
            mem_state = self.memup_module.init_state(len(full_batch.trajs))
            preds, targets, _ = self.memup_module.process_batch(full_batch, mem_state)
            all_preds.append(preds)
            all_targets.append(targets)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        self.memup_module.train()

        result = {m.name: m(all_preds, all_targets) for m in self.metrics}
        result['preds'] = all_preds.detach().cpu().numpy()
        result['targets'] = all_targets.detach().cpu().numpy()

        return result



