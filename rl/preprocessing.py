from datagen import TrajectoryPreprocessor, Trajectory
import numpy as np
from torch.nn import functional as F


class DiscountedReturn(TrajectoryPreprocessor):

    def __init__(self, gamma=0.99, key='return'):
        self.gamma = gamma
        self.key=key

    def process_single_(self, trajectory: Trajectory) -> Trajectory:
        traj_return = self.compute(trajectory.data['reward'], gamma=self.gamma)
        trajectory.data[self.key] = traj_return
        return trajectory

    @staticmethod
    def compute(rewards, next_value=0, masks=None, gamma=0.99):
        """
        Computes discounted n-step returns for rollout. Expects tensors or numpy.arrays as input parameters
        The function doesn't detach tensors, so you have to take care of the gradient flow by yourself.
        :return:
        """
        rollout_steps = len(rewards)
        returns = np.zeros(rollout_steps, dtype=np.float32) #[None] * rollout_steps
        if not masks:
            masks = [1.] * rollout_steps

        R = next_value
        for t in reversed(range(rollout_steps)):
            R = rewards[t] + gamma * masks[t] * R
            returns[t] = R
        return returns


class SumNextNRewards(TrajectoryPreprocessor):

    def __init__(self, num_rewards=5, key='reward_sum'):
        self.num_rewards=num_rewards
        self.key = key

    def process_single_(self, trajectory: Trajectory) -> Trajectory:
        R = trajectory.data['reward']
        n = self.num_rewards
        next_n_sum = [sum(R[i:i + n]) for i in range(len(R))]
        trajectory.data[self.key] = next_n_sum
        return trajectory


class AddPrevActionAndReward(TrajectoryPreprocessor):

    def __init__(self, num_actions=None):
        self.num_actions = num_actions

    def process_single_(self, trajectory: Trajectory) -> Trajectory:
        rewards = trajectory.data['reward']
        acts = trajectory.data['action']
        if len(acts.shape) == 1:
            acts = F.one_hot(acts, self.num_actions)

        prev_r = np.zeros(len(trajectory), dtype=np.float32)
        prev_r[1:] = rewards[:-1]

        prev_a = np.zeros((len(trajectory), self.num_actions), dtype=np.float32)
        prev_a[1:] = acts[:-1]

        trajectory.data['prev_action'] = prev_a
        trajectory.data['prev_reward'] = prev_r

        return trajectory