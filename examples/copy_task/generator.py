from datagen import TrajGenerator, Trajectory
import numpy as np


class CopyTaskGenerator(TrajGenerator):

    def __init__(self, head_size: int = 10, min_middle_size: int = 100, max_middle_size: int = None):
        self.head_size = head_size
        self.min_middle_size = min_middle_size
        self.max_middle_size = min_middle_size if max_middle_size is None else max_middle_size
        self._curr_episode_idx = 0

    def gen_trajectory(self):
        head = np.random.randint(0, 8, self.head_size)
        middle_size = np.random.randint(self.min_middle_size, self.max_middle_size+1)
        middle = np.zeros(middle_size, dtype=np.int64) + 8
        tail = np.zeros(self.head_size, dtype=np.int64) + 9

        x = np.concatenate([head, middle, tail])
        y = np.concatenate([head * 0 + 8, middle, head])

        self._curr_episode_idx += 1
        return Trajectory({"x": x, "y": y, 'traj_idx':self._curr_episode_idx})


class FiniteCopyTaskGenerator(CopyTaskGenerator):

    def __init__(self, n: int, head_size: int = 10, min_middle_size: int = 100, max_middle_size: int = None):
        super().__init__(head_size, min_middle_size, max_middle_size)
        self.n = n
        self.data = [super().gen_trajectory() for _ in range(n)]

    def gen_trajectory(self):
        return np.random.choice(self.data)


class PositionalCopyTaskGenerator(CopyTaskGenerator):

    def gen_trajectory(self):
        head = np.random.randint(0, 5, self.head_size)
        middle_size = np.random.randint(self.min_middle_size, self.max_middle_size + 1)
        middle = np.zeros(middle_size, dtype=np.int64) + 5
        tail = np.arange(self.head_size) + 6

        x = np.concatenate([head, middle, tail])
        y = np.concatenate([head * 0 + 5, middle, head])

        self._curr_episode_idx += 1
        return Trajectory({"x":x, "y":y, 'traj_idx':self._curr_episode_idx})
