from typing import List

from datagen import Trajectory
from memup import UncertaintyDetector
from .memup_trainer import load_model_from_generator_config
import torch
import numpy as np

class LegacyQRDQNDetector(UncertaintyDetector):
    """
    This class allows to check if new code works with old uncertainty detector.

    """
    def __init__(self, qrdqn_model, batch_size=1000):
        super(LegacyQRDQNDetector, self).__init__()
        self.model = qrdqn_model
        self.device = next(self.model.parameters()).device
        self.batch_size = batch_size

    @torch.no_grad()
    def make_estimate(self, trajectories: List[Trajectory]) -> List[List[float]]:
        uncertainties = []
        for t in trajectories:
            ep_unc = []
            obs = t.data['observation'][:-1]
            acts = t.data['action']
            returns = t.data['return']
            for t in range(0, len(t), self.batch_size):
                curr_unc = self.model.calculate_uncertainty(
                   self.adapt_np(obs[t:t+self.batch_size]),
                   self.adapt_acts(acts[t:t+self.batch_size]),
                   returns,
                   method='std'
                )
                ep_unc.extend(curr_unc)

            uncertainties.append(np.asarray(ep_unc))

        return uncertainties

    def adapt_np(self, obs_batch):
        return torch.tensor(obs_batch, dtype=torch.float, device=self.device)

    def adapt_acts(self, acts):
        #acts.argmax(1)
        return torch.tensor(acts, device=self.device)