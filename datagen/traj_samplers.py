from abc import abstractmethod, ABC
import numpy as np
import random as rnd
from .traj_buffer import TrajGenerator, TrajectoryPreprocessor, IdentityPreprocessor

from datagen.preprocessing import TrajectoryPreprocessor


class TrajectorySampler(ABC):

    @abstractmethod
    def sample(self, batch_size=1):
        """
        Sample a batch_size of trajectories
        """
        pass

    @abstractmethod
    def epoch(self, batch_size, num_batches):
        """
        Returns a generator that yields batches
        """
        pass


class RandomSampler(TrajectorySampler):
    """
    Randomly samples full trajectories from buffer.
    """
    def __init__(self, traj_buffer):
        super(RandomSampler, self).__init__()
        self.traj_buffer = traj_buffer

    def sample(self, batch_size=1):
        return self._raw_traj_sample(batch_size)

    #def on_sample(self, *preprocessors: TrajectoryPreprocessor):
    #    self.preprocessors.extend(list(preprocessors))

    def _raw_traj_sample(self, num_trajs):
        batch = np.random.choice(self.traj_buffer.get_data(), num_trajs)
        return batch

    def epoch(self, batch_size, num_batches):
        #num_batches = epoch_size // batch_size + int(epoch_size % batch_size != 0)

        for i in range(num_batches):
            yield self.sample(batch_size)


class DynamicSampler(TrajectorySampler):
    """
    DynamicSampler creates trajectories on the fly.
    i.e. if you need to sample 4 trajectories then
    DynamicSampler returns 4 new trajectories generated
    by traj_gen.
    """
    def __init__(self,
                 traj_gen: TrajGenerator,
                 traj_preprocessors: TrajectoryPreprocessor,
    ):
        self.traj_generator = traj_gen
        self.preprocessors = traj_preprocessors

    def sample(self, batch_size=1):
        trajs = self.traj_generator.gen_trajs(batch_size)
        return self.preprocessors.process_(trajs)

    def epoch(self, batch_size, num_batches):
        for i in range(num_batches):
            yield self.sample(batch_size)


class OrderedSampler(TrajectorySampler):
    """
    Samples trajectories in the order they are stored in the buffer.
    Before traversing a dataset can shuffle trajectories in the dataset.
    This allows the go through the dataset without sampling the same trajectory twice.
    The downside of this sampler is that some samples could consist of less trajectories
    than specified in the batch_size argument!
    """
    def __init__(self, traj_buffer, shuffle=True):
        super(OrderedSampler, self).__init__()
        self.traj_buffer = traj_buffer
        self.shuffle = shuffle
        self._idx = 0

    def sample(self, batch_size=1):
        if self._idx == 0 and self.shuffle:
            self.traj_buffer.shuffle()
        trajs = self._sample_from_idx(batch_size, self._idx)
        self._idx += batch_size
        if self._idx >= len(self.traj_buffer):
            self._idx = 0

        return trajs

    def _sample_from_idx(self, batch_size, start_idx):
        return self.traj_buffer.get_data()[start_idx: start_idx + batch_size]

    def epoch(self, batch_size, num_batches=None):
        assert num_batches is None, "num_batches is ignored for this sampler"
        if self.shuffle:
            self.traj_buffer.shuffle()

        for left_idx in range(0, len(self.traj_buffer), batch_size):
            yield self._sample_from_idx(batch_size, left_idx)

