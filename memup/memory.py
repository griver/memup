import torch
import numpy as np
from datagen import TrajectoryPreprocessor
from .batch import TrajectorySlicer, MemupBatch, Trajectory


class MemUPMemory(object):

    def __init__(self, memory_module):
        self.memory = memory_module
        self.preprocessor = TrajectorySlicer(memory_module.input_keys()
        )

    def process_batch(self, batch, mem_state):
        mem_input = self.preprocessor.process(
            batch.trajs, batch.mem_idx, self.device
        )

        output, mem_state = self.memory(mem_input, mem_state)
        filtered = self.filter_output_for_prediction(
            output, batch
        )

        return filtered, mem_state

    def filter_output_for_prediction(
            self,
            mem_output,
            batch
    ):
        res_ids = []
        L = mem_output.shape[1]
        shift = 0
        for idx, m_idx in zip(batch.pred_idx, batch.mem_idx):

            th_idx = torch.as_tensor(idx) - m_idx[0] + shift
            res_ids.append(th_idx)
            shift += L

        res_ids = torch.cat(res_ids).type(torch.long).to(self.device)

        return mem_output.flatten(0, 1)[res_ids]

    def init_state(self, *args, **kwargs):
        return self.memory.init_state(*args, **kwargs)

    def detach_state(self, *args, **kwargs):
        return self.memory.detach_state(*args, **kwargs)

    @property
    def device(self):
        return self.memory.device

    def eval(self):
        return self.memory.eval()

    def train(self):
        return self.memory.train()


class AccumulatedMemoryPreprocessor(TrajectoryPreprocessor):

    def __init__(self, memup_memory: MemUPMemory, key='mem_context'):
        self.memup_memory = memup_memory
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
        mem_state = self.memup_memory.init_state(len(trajectories))
        mem_output, _ = self.memup_memory.process_batch(full_batch, mem_state)
        context = torch.split(mem_output.cpu(), lengths)
        for c, t in zip(context, trajectories):
            t.data[self.key] = c

        return trajectories

    def process_single_(self, trajectory: Trajectory) -> Trajectory:
        pass

