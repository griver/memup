from abc import ABC, abstractmethod
from typing import Iterable, List, Dict
from copy import copy
import numpy as np
from .trajectory import Trajectory


class TrajectoryPreprocessor(ABC):
    """
    Class for all preprocessings that add and/or modify trajectory data
    For example:
          adds discounted total returns or n-step returns or GAE
          adds observations that are shifted by t-steps into future
          adds
    """

    @abstractmethod
    def process_single_(self, trajectory: Trajectory) -> Trajectory:
        pass

    def process_(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        for i in range(len(trajectories)):
            trajectories[i] = self.process_single_(trajectories[i])
        return trajectories

    def process_single(self, trajectory: Trajectory) -> Trajectory:
        return self.process_single_(copy(trajectory))

    def process(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        return self.process_([copy(t) for t in trajectories])


class Composite(TrajectoryPreprocessor):
    """
    Allows to compose several different preprocessors
    """
    def __init__(self, preprocessors):
        super(Composite, self).__init__()
        self.preprocessors = preprocessors

    def process_single_(self, trajectory: Trajectory) -> Trajectory:
        for p in self.preprocessors:
            trajectory = p.process_single_(trajectory)
        return trajectory

    def process_(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        for p in self.preprocessors:
            trajectories = p.process_(trajectories)
        return trajectories


class IdentityPreprocessor(TrajectoryPreprocessor):
    """
    Does nothing to a trajectory
    """
    def process_(self, trajectories):
        return trajectories

    def process(self, trajectories):
        return trajectories

    def process_single_(self, trajectory):
        return trajectory

    def process_single(self, trajectory):
        return trajectory


class AddDoneFlag(TrajectoryPreprocessor):
    """
    Adds an array of boolean values that signals the end of a trajectory.

    TerminalPrerpocessor is used in supervised setting as it helps
    to reset RNN hidden state at the end of a trajectory
    In RL setting trajectories are typically already contain a list of terminals
    """
    def process_single_(self, trajectory: Trajectory):
        if not trajectory.data.get('done', None):
            terminals = np.full(len(trajectory), False, dtype=np.bool)
            terminals[-1] = True
            trajectory.data['done'] = terminals
        return trajectory


class AddTailEvalTargets(TrajectoryPreprocessor):
    """
    Adds indices of the right prediction targets for the copy task
    in the traj.data['eval_target_key']
    This class is used to for memory evaluation for tasks
    like T-maze and Copy
    See datagen.batch_samplers.MemUPEvalSampler
    """
    def __init__(self,  tail_len, eval_target_key = 'eval_targets'):
        super(AddTailEvalTargets, self).__init__()
        self.eval_target_key = eval_target_key
        self.tail_len = tail_len

    def process_single_(self, trajectory: Trajectory) -> Trajectory:
        T = len(trajectory)
        left = max(T - self.tail_len, 0)
        trajectory.data[self.eval_target_key] = np.arange(left, T)
        return trajectory


class AddEvalTargets(TrajectoryPreprocessor):

    def __init__(self,  target_mask_key="m", eval_target_key="eval_targets"):
        super(AddEvalTargets, self).__init__()
        self.eval_target_key = eval_target_key
        self.target_mask_key = target_mask_key

    def process_single_(self, trajectory: Trajectory) -> Trajectory:
        m = trajectory.data[self.target_mask_key]
        trajectory.data[self.eval_target_key] = np.nonzero(m)[0]
        return trajectory


# class FrozenNeuralPreprocessor(TrajectoryPreprocessor):
#     """
#     Processes trajectories with some pretrained model and adds new key that contains model's output.
#     """
#
#     def __init__(self, model, models_preprocessor, key='neural_context'):
#         self.model = model
#         self.key = key
#         self.models_preprocessor = models_preprocessor
#
#     @torch.no_grad()
#     def process_single(self, trajectory: Trajectory) -> Trajectory:
#         data = self.models_preprocessor.process([trajectory])
#         hidden_states, _ = self.model.forward(data)
#         trajectory.data[self.key] = hidden_states[0]
#         return trajectory