import numpy as np

from .trajectory import Trajectory
from .preprocessing import  TrajectoryPreprocessor, IdentityPreprocessor
from abc import ABC, abstractmethod
from typing import Optional, List
from tqdm import tqdm

class TrajGenerator(ABC):
    """
    Base class for entities that generate trajectories.
    """
    @abstractmethod
    def gen_trajectory(self) -> Optional[Trajectory]:
        """
        :return: a new Trajectory or None (if no more trajectories can be generated)
        """
        raise NotImplementedError()

    def gen_trajs(self, up_to_n_trajs: int, verbose: bool=False) -> List[Trajectory]:
        trajs = []
        counter = range(up_to_n_trajs)
        if verbose:
            counter = tqdm(counter)

        for _ in counter:
            tr = self.gen_trajectory()
            if tr is None:
                break

            trajs.append(self.gen_trajectory())

        return trajs

    # def fill_buffer(self, traj_buffer, up_to_n: int) -> int:
    #     """
    #     Tries to upload up_to_n into a traj_buffer, but could possibly upload less if the generator can't produce more.
    #     :param traj_buffer: a TrajectoryBuffer
    #     :param up_to_n: maximum number of trajectories to upload in buffer
    #     :return: number of actually uploaded trajectories
    #     """
    #     for i in range(up_to_n):
    #         traj = self.gen_trajectory()
    #         if traj is None:
    #             break
    #
    #         traj_buffer.add_trajectory(traj)
    #
    #     return i+1


class TrajectoryBuffer:
    """
    Acts as a list of trajectories but different in two main aspects:
    1) each trajectory is preprocessed with static_preprocessor at the moment when it is added in the buffer
    2) you can update preprocessing for all or some trajectories by calling update_preprocessing
    """
    def __init__(
            self,
            maxlen: int,
            static_preprocessor: TrajectoryPreprocessor=IdentityPreprocessor()
    ):
        self.maxlen = maxlen
        self.next_idx = 0
        self.is_full = False
        self._buffer = np.empty(maxlen, dtype=object)
        #static preprocessor processes a trajectory only once when it is added to the buffer
        self.static_preprocessor = static_preprocessor
        #self.trajgen = trajectory_generator


    def add_trajectory(self, tr: Trajectory):
        tr = self.static_preprocessor.process_single_(tr)
        self._buffer[self.next_idx] = tr
        if self.next_idx == self.maxlen - 1:
            self.is_full = True
        self.next_idx = (self.next_idx + 1) % self.maxlen

    def add_trajectories(self, trajectories: List[Trajectory]):
        for tr in trajectories:
            self.add_trajectory(tr)

    def update_preprocessing(self, indices=None):
        if indices is None:
            indices = range(len(self))

        for i in indices:
            self._buffer[i] = self.static_preprocessor.process_single_(self._buffer[i])

    def __len__(self):
        return self.maxlen if self.is_full else self.next_idx

    def __getitem__(self, item):
        return self._buffer.__getitem__(item)

    def shuffle(self):
        np.random.shuffle(self._buffer[0:self.__len__()])

    def get_data(self):
        return self._buffer[0:self.__len__()]

    #def to_list(self):
    #    return list(self.buffer)