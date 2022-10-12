from typing import Iterable, Optional

from datagen.preprocessing import IdentityPreprocessor, TrajectoryPreprocessor
from datagen import TrajectoryBuffer, TrajectorySampler

from memup.target_selector import TargetSelector
from memup.batch import MemupBatch

from copy import copy
import numpy as np
from abc import ABC, abstractmethod
from typing import Iterator

class MemUPBatchSampler(ABC):

    @abstractmethod
    def sample(self, batch_size: int=1) -> MemupBatch:
        pass

    @abstractmethod
    def epoch(self, batch_size, num_batches) -> Iterator[MemupBatch]:
        pass

    @abstractmethod
    def make_batch_from(self, trjectories, prediction_freq=1) -> MemupBatch:
        pass

class FullTrajMemUPSampler(MemUPBatchSampler):
    """
    Processes batches of raw episodes to add prediction targets
    and change data format that is understood by the MemUP modules
    """
    def __init__(
            self,
            traj_sampler: TrajectorySampler,
            #uncertainty_detector: UncertaintyDetector,
            target_selector: TargetSelector,
            dynamic_preprocessor=None,
            prediction_frequency=1,
            relative_prediction_indices=True

    ):
        super().__init__()
        self.traj_sampler = traj_sampler
        self.prediction_freq = prediction_frequency
        self.target_selector = target_selector
        self.relative_prediction_indices = relative_prediction_indices
        # dynamic_preprocessor processes a trajectory each time it is sampled from buffer
        # for preprocessing that can be done only once see TrajectoryBuffer.static_preprocessor
        if dynamic_preprocessor is None:
            dynamic_preprocessor = IdentityPreprocessor()
        self.dynamic_preprocessor = dynamic_preprocessor

    def sample(self, batch_size=1):
        return self.make_batch_from(
            self.traj_sampler.sample(batch_size),
            self.prediction_freq
        )

    def epoch(self, batch_size, num_batches) -> Iterator[MemupBatch]:
        for trajs in self.traj_sampler.epoch(batch_size, num_batches):
            yield self.make_batch_from(trajs, self.prediction_freq)

    def make_batch_from(self, trajectories, prediction_freq=1):
        """
        This is RL specific method, for SL you'll probably
        need to rewrite it along with changing create_rl_targets
        """
        #unc_estimates = self.uncertainty_detector.make_estimate(trajs)
        trajectories = self.dynamic_preprocessor.process(trajectories)
        batch = MemupBatch([], [], [], [])

        for i, e in enumerate(trajectories):
            pred_idx = self._prediction_idx(0, len(e), prediction_freq)
            target_idx = self.target_selector.select_target_idx(e, pred_idx)

            batch.mem_idx.append(np.arange(len(e)))
            batch.pred_idx.append(pred_idx)
            batch.target_idx.append(target_idx)
            batch.trajs.append(e)

        return batch

    def _prediction_idx(self, start, exclusive_end, prediction_freq):
        """
        Returns every i-th index in a given range(start, exclusive_end),
        where i is given by prediction_freq.
        It also never excludes the last step
        For example:
              input: (25, 35, 5), output: [29, 34]
              input: (25, 35, 7), output: [31, 34]
        :return:
        """
        if self.relative_prediction_indices:
            raise NotImplementedError()

        idx = list(np.arange(start, exclusive_end + 1, prediction_freq)[1:] - 1)
        if (exclusive_end - 1) not in idx[-1:]:
           idx.append(exclusive_end - 1)
        return idx


class TruncatedMemUPSampler(FullTrajMemUPSampler):
    """
        Instead of batches with full episodes this class generates
        batches of episode slices, i.e.:
            full_batch = [ep1[1:1000], e2[1:1000], .., eN[1:1000]]
            will become a sequence of batches:
                [ep1[1:200],.., epN[1:200]],

                [ep1[200:400],.., epN[200:400]],
                ....
                [ep1[800:1000],.., epN[800:1000]]
            ]
        """
    def __init__(
            self,
            traj_sampler: TrajectorySampler,
            target_selector: TargetSelector,
            dynamic_preprocessor=None,
            prediction_frequency = 1,
            relative_prediction_indices=False,
            rollout = 1,
    ):
        super().__init__(
            traj_sampler,
            target_selector,
            dynamic_preprocessor,
            prediction_frequency,
            relative_prediction_indices
        )
        assert rollout >= 1, 'Rollout should be a positive integer'
        self.rollout = rollout

        self._mem_offset = None
        self._ep_lengths = None
        self._batch_buffer = None
        self._batch_size = None

    def sample(self, batch_size=1):
        self._init_buffers(batch_size)
        self._fill_batch()
        return self._next_rollout()

    def epoch(self, batch_size, num_batches) -> Iterable[MemupBatch]:
        """
        Returns a generator that yields randomly sampled batches.
        """
        self._init_buffers(batch_size)

        for i in range(num_batches):
            # if some of previous episodes is finished
            # then fill episode buffer with new episodes
            self._fill_batch()
            yield self._next_rollout()

    def _init_buffers(self, batch_size):
        update_buffers = False
        if self._batch_size is None:
            update_buffers = True

        elif self._batch_size != batch_size:
            all_done = np.all(self._mem_offset >= self._ep_lengths)
            if not all_done:
                raise ValueError(
                    'You are trying to change batch_size while some episodes are still hanging'
                )
            update_buffers = True

        if update_buffers:
            self._batch_size = batch_size
            self._mem_offset = np.zeros(batch_size, dtype=np.int)
            self._ep_lengths = np.zeros(batch_size, dtype=np.int)
            self._batch_buffer = None

    def _fill_batch(self):
        """
        If episodes have different lengths, then some of them could have
        ended during previous truncated slice.
        Therefore we need to put new episodes where the old one are ended/
        """
        done_episodes = self._mem_offset >= self._ep_lengths
        done_ids = done_episodes.nonzero()[0]
        if len(done_ids) == 0: return  # no finished episodes

        new_trajs = self.traj_sampler.sample(len(done_ids))
        new_length = [len(e) for e in new_trajs]

        new_batch = self.make_batch_from(new_trajs, 1)

        self._ep_lengths[done_ids] = new_length
        self._mem_offset[done_ids] = 0

        if len(done_ids) == self._batch_size:
            # if all episodes is new then just replace the buffer
            self._batch_buffer = new_batch
            return

        self._batch_buffer.update_indices(done_ids, new_batch)

    def _next_rollout(self):
        """
        This class tracks where previous rollout has ended and
        this function returns the next slice of steps for each
        episodes.
        """
        batch = copy(self._batch_buffer)
        for i in range(self._batch_size):
            m_start = self._mem_offset[i]
            m_end = min(m_start + self.rollout, len(batch.trajs[i]))

            batch.mem_idx[i] = np.arange(m_start, m_end)
            preds = self._prediction_idx(m_start, m_end, self.prediction_freq)
            batch.target_idx[i] = [batch.target_idx[i][t] for t in preds]
            batch.pred_idx[i] = preds

        self._mem_offset += self.rollout

        return batch


class MemUPEvalSampler(MemUPBatchSampler):

    """
    Evaluating memory trained with MemUP is tricky:
    1. you need to test predictions of memory+predictor at steps where temporal dependencies end. Selection of these steps should be hard-coded as trained uncertainty detector may fail at selecting the right steps.

    2. To make sure that memory module can store relevant information till the right moment, we need to use memory states from the same hard-coded steps, i.e. if we predict $y_i$ then we use $m_i$ (not some $m_j$ where j < i).

    Therefore, it is assumed that these hard-coded steps used are stored in trajectory.data[eval_target_key].
    Given these hard-coded steps MemUPEvalSampler produce target_idx and pred_idx that both contain indices of these hard-coded steps.

    """

    def __init__(
            self,
            traj_sampler: TrajectorySampler,
            eval_target_key: str='eval_targets',
            dynamic_preprocessor=None
    ):
        super().__init__()
        self.traj_sampler = traj_sampler
        self.eval_target_key = eval_target_key
        # dynamic_preprocessor processes a trajectory each time it is sampled from buffer
        # for preprocessing that can be done only once see TrajectoryBuffer.static_preprocessor
        if dynamic_preprocessor is None:
            dynamic_preprocessor = IdentityPreprocessor()
        self.dynamic_preprocessor = dynamic_preprocessor
        self._check_eval_trajs()


    def _check_eval_trajs(self):
        """
        Checks if trajectories in buffer have eval_target_key
        :return: None if everything is correct, but rises exception otherwise
        """
        traj = self.traj_sampler.sample()[0]
        traj = self.dynamic_preprocessor.process([traj])[0]
        if self.eval_target_key not in traj.data.keys():
            raise ValueError(
               f'Trajectories provided by {type(self.traj_sampler)} do not have {self.eval_target_key} required for evaluation'
           )

    def make_batch_from(self, trjectories, prediction_freq=None) -> MemupBatch:

        trjectories = self.dynamic_preprocessor.process(trjectories)
        batch = MemupBatch([], [], [], [])

        for i, tr in enumerate(trjectories):
            eval_steps_idx = np.asarray(tr.data[self.eval_target_key])

            batch.mem_idx.append(np.arange(len(tr)))
            batch.pred_idx.append(eval_steps_idx)
            batch.target_idx.append(eval_steps_idx.reshape(-1,1))
            batch.trajs.append(tr)

        return batch

    def sample(self, batch_size=1):
        """
        Sample a single batch randomly
        """
        trajs = self.traj_sampler.sample(batch_size)
        return self.make_batch_from(trajs)

    def epoch(self, batch_size, num_batches=None):

        for trajs in self.traj_sampler.epoch(batch_size, num_batches):
            yield self.make_batch_from(trajs)


class Seq2SeqMemUPEvalSampler(MemUPEvalSampler):

    def __init__(
            self,
            traj_sampler: TrajectorySampler,
            eval_pred_key: str = 'eval_preds',
            eval_target_key: str = 'eval_targets',
            dynamic_preprocessor=None
    ):
        super().__init__(traj_sampler, "", dynamic_preprocessor)
        self.eval_pred_key = eval_pred_key
        self.eval_target_key = eval_target_key

    def _check_eval_trajs(self):
        pass

    def make_batch_from(self, trjectories, prediction_freq=None) -> MemupBatch:

        trjectories = self.dynamic_preprocessor.process(trjectories)
        batch = MemupBatch([], [], [], [])

        for i, tr in enumerate(trjectories):
            eval_pred_idx = np.asarray(tr.data[self.eval_pred_key])
            eval_target_idx = np.asarray(tr.data[self.eval_target_key])

            batch.mem_idx.append(np.arange(len(tr)))
            batch.pred_idx.append(eval_pred_idx)
            batch.target_idx.append(eval_target_idx.reshape(-1, 1))
            batch.trajs.append(tr)

        return batch
