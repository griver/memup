import torch
import numpy as np
import torch.distributions.categorical as th_categorical
from abc import ABC, abstractmethod
from datagen import TrajectoryPreprocessor, Trajectory
import logging

class TargetSelector(object):
    """
    Selects indices of high uncertainty steps in the trajectory based on
    uncertainty estimates H(f_t|c_t) of each trajectory step
    """
    def __init__(self,  k, time_dependent_selection=True, uncertainty_key='uncertainty'):
        """
        :param k: how much future steps with high uncertainty we want to predict
        :param time_dependent_selection: if True then for prediction moment t selector will choose only high uncertainty steps
         that happened after the moment t.
        """
        self.k = k
        self.time_dependent_selection = time_dependent_selection
        self.uncertainty_key = uncertainty_key

    @abstractmethod
    def target_idx_from_estimate(self, uncertainty_estimate, mem_prediction_steps=None) :
        """
        Returns indices of target steps for each step in mem_prediction_steps
        if mem_prediction_steps is None, then we consider all prediction steps
        :param mem_prediction_steps: uncertainty estimates for each step in a trajectory
        :param prediction_steps: steps from which the memory state will be used to make long-term future predictions
        :return: result = [list_of_future_targets1, list_of_future_targets2]; len(result) == len(mem_prediction_steps)
        """
        pass

    def select_target_idx(self, trajectory, mem_prediction_steps=None):
        if mem_prediction_steps is None:
            mem_prediction_steps = np.arange(len(trajectory))

        return self.target_idx_from_estimate(
            trajectory.data[self.uncertainty_key],
            mem_prediction_steps
        )


class AddEvalTargetsFromTargetSelector(TrajectoryPreprocessor):
    """
    This preprocessor allows
    """
    def __init__(
        self,
        target_selector: TargetSelector,
        eval_target_key='eval_targets'
    ):
        if target_selector.time_dependent_selection is True:
            logging.warning(
                'AddEvalTargetsFromTargetSelector will use only'
                ' targets selected for the first step in the trajectory'
            )

        self.target_selector = target_selector
        self.eval_target_key = eval_target_key

    def process_single_(self, trajectory: Trajectory) -> Trajectory:
        eval_targets = self.target_selector.select_target_idx(trajectory, [0])
        trajectory.data[self.eval_target_key] = eval_targets[0]
        return trajectory


class TopKSelector(TargetSelector):
    """
    Selects 'k' indices from the array 'a' that correspond to the highest values in 'a'.
    If the number of remaining elements in the sequence is less than k,
    then some randomly selected elements will be repeated.
    """
    def target_idx_from_estimate(self, uncertainty_estimates, mem_prediction_steps=None):
        a = np.asarray(uncertainty_estimates)
        if mem_prediction_steps is None:
            mem_prediction_steps = np.arange(len(a))
        n = len(a)
        assert n >= self.k, 'Episode length is smaller than number of events to select!'

        top_idx = np.argsort(a)
        top_k = set(top_idx[-self.k:])
        if self.time_dependent_selection is False: #absolute top values for the whole trajectory
            return [list(top_k) for t in mem_prediction_steps]

        next_candidate = n - self.k - 1
        top_k_list = []

        for i in mem_prediction_steps:
            top_k.difference_update({e for e in top_k if e < i})

            while next_candidate >= 0 and len(top_k) < self.k:
                if top_idx[next_candidate] >= i:
                    top_k.add(top_idx[next_candidate])
                next_candidate -= 1

            elements = list(top_k)
            if len(elements) < self.k: #if there are not enouth values we repeat some of them:
                elements.extend(np.random.choice(elements, self.k-len(elements)))
            top_k_list.append(elements)

        return top_k_list


class CurrentStepSelector(TargetSelector):
    """
    Instead of selecting arbitrary distant future events based on values from a,
    CurrentSelector just selects the same step,
    e.g. from memory state at step 7 we will predict target variable at step 7.
    """
    def __init__(self, k=1, time_dependent_selection=True):
        super().__init__(k, time_dependent_selection)
        assert self.k == 1, "This method only works when k==1!"
        assert time_dependent_selection == True, "we predict y[i] for  don't make sense with next step prediction"

    def target_idx_from_estimate(self, uncertainty_estimate, mem_prediction_steps=None):
        if mem_prediction_steps is None:
            mem_prediction_steps = np.arange(len(uncertainty_estimate)).reshape(-1, 1)
        return np.asarray(mem_prediction_steps).reshape(-1, 1)


class CompositeSelector(TargetSelector):
    """
    MixedSelector allows to
    """
    def __init__(self, selectors):
        total_k = sum(s.k for s in selectors)
        super().__init__(total_k, None)
        self.selectors = selectors


    def target_idx_from_estimate(self, uncertainty_estimate, mem_prediction_steps=None):
        targets = lambda ts: ts.target_idx_from_estimate(uncertainty_estimate, mem_prediction_steps)
        return [np.concatenate(e) for e in zip(*[targets(ts) for ts in self.selectors])]


class TemperatureSamplingSelector(TargetSelector):
    """
    Selects targets steps by sampling proportionally to uncertainty values.
    if time_dependent_selection is True:
        sampling is implemented with temperature softmax
    otherwise:
        sampling is implemented via gumbel-max trick
    """
    def __init__(
            self, k,
            temperature=0.25,
            time_dependent_selection=True,
            uncertainty_key='uncertainty',
            resample_every=20,
    ):
        """
        :param k: num events to select
        :param temperature: higher temperature means more uniform sampling
        :param time_dependent_selection:
        :param uncertainty_key:
        :param resample_every: how often to recompute gumbel-max noise.
               resample_every is used only when time_dependent_selection is False
        """
        super().__init__(k, time_dependent_selection, uncertainty_key)
        self.temperature = temperature
        self.resample_every = resample_every

    @torch.no_grad()
    def target_idx_from_estimate(self, uncertainty_estimate, mem_prediction_steps=None):
        temp_a = torch.as_tensor(uncertainty_estimate)/self.temperature
        T = len(temp_a)

        if self.time_dependent_selection is False:
            softmax = th_categorical.Categorical(logits=temp_a)
            global_selection = softmax.sample([self.k]).numpy()
            return [global_selection] * len(mem_prediction_steps) #[rnd_k]

        sampled_k_list = []
        for i, t in enumerate(mem_prediction_steps):
            if i % self.resample_every == 0:
                noisy_a = self.add_gumbel_noise(temp_a)

            if T - t >= self.k:
                sampled_k = torch.topk(noisy_a[t:], self.k).indices.numpy() + t
            else:
                sampled_k = list(range(t, T))
                sampled_k.extend(np.random.choice(sampled_k, self.k - len(sampled_k)))

            sampled_k_list.append(sampled_k)

        return sampled_k_list

    def gumbel_noise(self, shape, eps=1e-20):
        noise = torch.rand(shape)
        return -torch.log(-torch.log(noise + eps) + eps)

    def add_gumbel_noise(self, logits, eps=1e-20):
        return logits + self.gumbel_noise(logits.shape, eps)


#TODO: Make classes from these functions
def rnd_future_k(a, k, per_episode_targets=True, ignore_first_t=0):
    indices = list(np.arange(len(a)))
    left = max(0, ignore_first_t)

    #sampling without replacement complicates cases when k > 1:
    rnd_k = np.random.choice(indices[left:],k)
    if per_episode_targets:
        return [rnd_k]

    rnd_k_list = []
    for i in range(len(a)):
        left = max(left, i)
        rnd_k = np.random.choice(indices[left:], k)
        rnd_k_list.append(rnd_k)

    return rnd_k_list

