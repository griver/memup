from collections import defaultdict
from typing import List

import torch
import numpy as np

from datagen import Trajectory


class MemupBatch(object):

    def __init__(self, trajs, mem_idx, pred_idx, target_idx):
        """
        :param trajs: full trajectories with all additional data
        :param mem_idx: indices of current rollout
        :param pred_idx: indices in current rollout from which prediction is run
        :param target_idx: indices of steps that we want to predict
        """
        self.trajs: List[Trajectory] = trajs
        self.mem_idx = mem_idx
        self.pred_idx = pred_idx
        self.target_idx = target_idx

    def _assign(self, trajs, mem_idx, pred_idx, target_idx):
        self.trajs = trajs
        self.mem_idx = mem_idx
        self.pred_idx = pred_idx
        self.target_idx = target_idx

    def __copy__(self):
        return MemupBatch(
            list(self.trajs),
            list(self.mem_idx),
            list(self.pred_idx),
            list(self.target_idx),
        )

    def __getitem__(self, item):
        return MemupBatch(
            self.trajs.__getitem__(item),
            self.mem_idx.__getitem__(item),
            self.pred_idx.__getitem__(item),
            self.target_idx.__getitem__(item)
        )

    def update_indices(self, indices, batch):
        for new_id, my_id in  enumerate(indices):
            self.trajs[my_id] = batch.trajs[new_id]
            self.mem_idx[my_id] = batch.mem_idx[new_id]
            self.pred_idx[my_id] = batch.pred_idx[new_id]
            self.target_idx[my_id] = batch.target_idx[new_id]


# class BatchPreprocessor(ABC):
#     """
#     Abstract class for all classes that will prepare data from MemupBatch
#     to be consumed by neural networks.
#     """
#     @abstractmethod
#     def process(self, batch_data, device: torch.device):
#         pass


class TrajectorySlicer(object): #(BatchPreprocessor):
    """
    Selects data from the list of trajectories
    based on received indices.
    """
    def __init__(self, input_keys):
        """
        :param input_keys: which keys in the trajectory to process
        """
        self.input_keys = set(input_keys)

    def process(self, trajs, indices=None, device="cpu"):
        """
        Slices trajectory data that stored in self.input_keys
        and converts it to torch.Tensors on a specified device.
        Example:
        ```
        mem_net = LstmMemory(...)
        slicer = TrajectorySlicer(mem_net.input_keys())
        input_dict = slicer.process(batch.trajs, batch.mem_idx)
        ... = mem_net(input_dict, ...)

        ```

        :param trajs: list of trajectories
        :param indices: a list that stores indices for each trajectory
        :param device: torch.device or string
        :return: a dict with keys from self.input_keys and values with lists of torch.tensors
        """
        if indices is None:
            indices = [None]*len(trajs)

        input_data = defaultdict(list)
        for idx, traj in zip(indices, trajs):
            for k in self.input_keys:
                input_data[k].append(
                    slice_to_torch(traj.data[k], idx, "cpu")
                )

        for k in self.input_keys:
            lens = [t.shape[0] for t in input_data[k]]
            flat = torch.cat(input_data[k]).to(device)
            input_data[k] = torch.split(flat, lens)

        return input_data

    # def process(self, trajs, indices=None, device="cpu"):
    #     """
    #     Slices trajectory data that stored in self.input_keys
    #     and converts it to torch.Tensors on a specified device.
    #     Example:
    #     ```
    #     mem_net = LstmMemory(...)
    #     slicer = TrajectorySlicer(mem_net.input_keys())
    #     input_dict = slicer.process(batch.trajs, batch.mem_idx)
    #     ... = mem_net(input_dict, ...)
    #     ```
    #     :param trajs: list of trajectories
    #     :param indices: a list that stores indices for each trajectory
    #     :param device: torch.device or string
    #     :return: a dict with keys from self.input_keys and values with lists of torch.tensors
    #     """
    #     if indices is None:
    #         indices = [None]*len(trajs)
    #
    #     input_data = defaultdict(list)
    #     for idx, traj in zip(indices, trajs):
    #         for k in self.input_keys:
    #             input_data[k].append(
    #                 slice_to_torch(traj.data[k], idx, device)
    #             )
    #
    #     return input_data

def slice_to_torch(seq, indices, device):
    array = np.asarray(seq)
    if indices is not None:
        array = array[np.asarray(indices)]
    #dtype = th_type(seq_slice)
    return torch.as_tensor(array, device=device)#, dtype=dtype)


# def th_type(array):
#     dtype=array.dtype
#     if np.issubdtype(dtype, np.floating):
#         return torch.float32
#     elif np.issubdtype(dtype, np.integer):
#         return torch.uint8
#     elif dtype == bool:
#         return torch.bool
#     else:
#         raise ValueError('meet unexpected dtype={}'.format(dtype))