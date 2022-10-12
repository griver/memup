import rl.experiment_utils as exp_utils
import torch
from torch.utils import tensorboard
import numpy as np
import os
import time
from collections import deque
from typing import Optional, Callable, Any, Dict, List
from memup import MemUPBatchSampler, MemUPEvalSampler, MemUPMemory, MemUPPredictor
from memup import training


class RLMemSummary(object):
    def __init__(self, savedir, eval_metric_name='MSE'):
        self.summary = tensorboard.SummaryWriter(os.path.join(savedir, 'summary'))
        self.train_accuracies = deque(maxlen=50)
        self.train_diffs = deque(maxlen=50)

        self.train_diffs.append(10.)
        self.train_accuracies.append(0.)

        self.eval_metric_name=eval_metric_name
        self.eval_diff = 0
        self.eval_accuracy = 0
        self.eval_metric = 0
        self.best_eval_metric = 0
        self.num_batches = 0
        self.num_env_steps = 0

    def add_train_step(self, train_info, batch=None):
        loss  = train_info['loss']
        self.num_batches += 1
        self.num_env_steps += sum(len(m) for m in batch.mem_idx)
        diff, accuracy = self._get_diff_and_acc(train_info)

        self.train_accuracies.append(accuracy)
        self.train_diffs.append(diff)

        self.summary.add_scalar('train/loss', loss, self.num_batches)
        self.summary.add_scalar('train/accuracy(0.3)', accuracy, self.num_batches)
        self.summary.add_scalar('train/loss-per-frames', loss, self.num_env_steps)
        self.summary.add_scalar('train/accuracy-per-frames', accuracy, self.num_env_steps)

    def add_eval_step(self, eval_info):
        eval_metric = eval_info[self.eval_metric_name]
        diff, accuracy = self._get_diff_and_acc(eval_info)

        self.eval_metric = eval_metric
        self.eval_diff = diff
        self.eval_accuracy = accuracy

        self.summary.add_scalar('eval/metric', eval_metric, self.num_batches)
        self.summary.add_scalar('eval/diff', diff, self.num_batches)
        self.summary.add_scalar('eval/accuracy', accuracy,self.num_batches)
        self.summary.add_scalar('eval/metric-per-frames', eval_metric, self.num_env_steps)
        self.summary.add_scalar('eval/diff-per-frames', diff, self.num_env_steps)
        self.summary.add_scalar('eval/accuracy-per-frames',  accuracy,self.num_env_steps)

    def print_info(self):
        print(f"Eval metric: {self.eval_metric:.4f}, diff: {self.eval_diff:.2f}, acc: {self.eval_accuracy:.3f}")

        print(f'Train diff: {np.mean(self.train_diffs):.2f}, acc: {np.mean(self.train_accuracies):.3f}')

    def _get_diff_and_acc(self, info_dict):
        preds = info_dict['preds']
        targets = info_dict['targets']
        diff = self.mse_diff(preds, targets)
        accuracy = self.mse_accuracy(preds, targets)
        return diff, accuracy

    @staticmethod
    def mse_diff(preds, targets):
        return np.abs(preds - targets).mean()

    @staticmethod
    def mse_accuracy(preds, targets, precision=0.3):
        is_close_enough = np.abs(preds - targets) < precision
        return is_close_enough.mean()

    def close(self):
        self.summary.close()


class RLMemUPTrainer(object):

    def __init__(
        self,
        config: Dict[str, Any],
        train_sampler: MemUPBatchSampler,
        memory: MemUPMemory,
        predictor: MemUPPredictor,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        summary: RLMemSummary,
        eval_sampler: Optional[MemUPEvalSampler]=None,
        eval_criterions: List[Callable]=None,
        update_callbacks: List[Callable]=tuple(),
        #memory_accumulator: Optional[nets.Network]=None,
        #predictor_accumulator: Optional[nets.Network]=None,
        save_frequency=1,
        min_loss_threshold=0.00002,
        verbose=True

    ):
        super(RLMemUPTrainer, self).__init__()

        self.config = config
        self.train_sampler = train_sampler
        self.eval_sampler = eval_sampler
        self.memory = memory
        self.predictor = predictor
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary = summary
        self.save_frequency = save_frequency
        self.update_callbacks = list(update_callbacks)
        self.min_loss_threshold = min_loss_threshold
        self.verbose = verbose

        if eval_criterions:
            self.eval_criterions = list(eval_criterions)
        else:
            self.eval_criterions = [criterion]

        self.initialized = False
        self.curr_epoch = 0
        self.curr_updates = 0
        self.curr_env_steps = 0
        #self.memory_accumulator = memory_accumulator
        #self.predictor_accumulator = predictor_accumulator
        self.savepath = os.path.join(config['logdir'], 'checkpoint/model.pth')
        self.savepath_best = os.path.join(config['logdir'], 'checkpoint/best_model.pth')

        self.reset()


    def reset(self):
        self.curr_epoch = 0
        self.curr_updates = 0

        self.mem_state = self.memory.init_state(self.config['batch_size'])
        self.initialized = True

    def train_n_epochs(self, num_epochs=1, eval=True):
        for _ in range(num_epochs):
            self.train_epoch()
            if eval:
                eval_info = self.evaluate()

            if self.curr_epoch % self.save_frequency == 0:
                self.save_model()

    # def _update_accumulators(self):
    #     if self.memory_accumulator:
    #         self.memory_accumulator.accumulate(self.memory.memory)
    #     if self.predictor_accumulator:
    #         self.predictor_accumulator.accumulate(self.predictor.predictor)

    def train_epoch(self, num_batches=None):
        self.msg(
            "========== EPOCH #{} =========".format(self.curr_epoch)
        )
        batch_size = self.config['batch_size']

        if not num_batches:
            num_batches = self.config['num_batches']

        start_time = time.time()
        epoch_batches = 0
        total_loss = 0.

        for batch in self.train_sampler.epoch(batch_size, num_batches):
            epoch_batches += 1

            update_info = training.update_on_batch(
                batch,
                self.memory,
                self.predictor,
                self.mem_state,
                self.criterion,
                self.optimizer,
                min_loss_threshold=self.min_loss_threshold
            )
            total_loss += update_info['loss']
            self.mem_state = update_info['mem_state']

            self._exec_callbacks()

            self.summary.add_train_step(update_info, batch)

            self.msg(
                '\rEpoch #{}, loss: {:.4f}, num_batches: {}'.format(
                    self.curr_epoch, total_loss / epoch_batches, epoch_batches), end=''
            )

        self.curr_epoch += 1
        time_diff = time.time() - start_time
        self.msg(f"\nUPS: {epoch_batches/time_diff:.2f}")


    def _exec_callbacks(self):
        for c in self.update_callbacks: c()

    def evaluate(self):
        if self.eval_sampler is None:
            return

        eval_info = training.eval_memory_and_predictor(
            self.eval_sampler,
            self.memory,
            self.predictor,
            self.eval_criterions,
            batch_size=16
        )
        self.summary.add_eval_step(eval_info)

        if self.verbose:
            self.summary.print_info()

        return eval_info

    def save_model(self, is_best=False):
        """ Save state in filename. Also save in best_filename if is_best. """
        model = {
            'memory':self.memory.memory.state_dict(),
            'predictor':self.predictor.predictor.state_dict(),
            'num_updates':self.curr_updates,
            'num_env_steps': self.curr_env_steps,
        }

        exp_utils.ensure_dir(self.savepath)
        torch.save(model, self.savepath)
        if is_best:
            exp_utils.ensure_dir(self.savepath_best)
            torch.save(model, self.savepath_best)
        #self.msg(f'saved model in {self.savepath}')

    def msg(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def close(self):
        self.summary.close()


def load_model_from_generator_config(
        config, obs_space, num_actions
):
    train_config = exp_utils.get_train_config(
        config['uncertainty_model']
    )
    model = exp_utils.load_qrdqn(
        config['uncertainty_model'], obs_space,
        num_actions, config['device'], train_config
    )
    return model
