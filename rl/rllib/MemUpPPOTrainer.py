import logging
from typing import Optional, Type, Dict, Any, Callable
import torch
from ray.rllib.agents.dreamer.dreamer import total_sampled_timesteps
from ray.rllib.agents.ppo import PPOConfig, PPOTrainer, PPOTorchPolicy
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.util.debug import log_once
from ray.rllib.agents.trainer import Trainer
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
)
from torch import nn
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import TrainerConfigDict, ResultDict
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    WORKER_UPDATE_TIMER,
)

from datagen import TrajectoryBuffer, Trajectory
from datagen.preprocessing import Composite, AddTailEvalTargets
from memup.nets import accumulate
from memup.training import MemupTrainOp, MemupModule, MemupEvalOp
from metrics import MSEMetric
from rl.training_utils import disable_gradients
from rl.preprocessing import DiscountedReturn
from torch.utils import tensorboard
import os

logger = logging.getLogger(__name__)


def create_cuda_module(create_module: Callable[[], nn.Module]):
    def create():
        return create_module().cuda()
    return create


class MemupPPOConfig(PPOConfig):

    def __init__(self):
        """Initializes a PPOConfig instance."""
        super().__init__()
        self.memup_train_every = 1
        self.memup = {
            "buffer_length": 1000,
            "eval_buffer_length": 100,
            "gamma": 0,
            "prefill_timesteps": 100,
            "create_predictor": None,
            "prediction_frequency": 1,
            "rollout_length": 10,
            "policy_accumulation_decay": 0.9,
            "batch_size": 128,
            "train_steps": 200
        }

    @override(PPOConfig)
    def training(
        self,
        *,
        memup: Dict[str, Any],
        **kwargs,
    ) -> "MemupPPOConfig":
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)
        self.memup = memup

        return self

class RLPolicySummary(object):
    def __init__(self, savedir):
        self.summary = tensorboard.SummaryWriter(os.path.join(savedir, 'summary'))
        self.num_updates = 0
        self.num_env_steps = 0

    def add_training_step(self, train_info):
        self.num_updates += 1
        self.num_env_steps = train_info['num_agent_steps_trained']
        mean_r = train_info['episode_reward_mean']
        self.summary.add_scalar('train/reward', mean_r, self.num_updates)
        self.summary.add_scalar('train/reward-per_frames', mean_r, self.num_env_steps)


    def add_eval_step(self, eval_info):
        mean_r = eval_info['evaluation']['episode_reward_mean']
        self.summary.add_scalar('eval/reward', mean_r, self.num_updates)
        self.summary.add_scalar('eval/reward-per-frames', mean_r, self.num_env_steps)

    def close(self):
        self.summary.close()

class MemupPPOTrainer(PPOTrainer):
    @classmethod
    @override(Trainer)
    def get_default_config(cls) -> TrainerConfigDict:
        return MemupPPOConfig().to_dict()

    @override(Trainer)
    def validate_config(self, config: TrainerConfigDict) -> None:
        super().validate_config(config)

    @override(Trainer)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            return PPOTorchPolicy
        else:
            raise NotImplementedError()

    def sync_memory(self):
        mem_net = self.get_policy().model.lstm
        accumulate(self.memup_module.memory_net, mem_net, self.config["memup"]["policy_accumulation_decay"])
        disable_gradients(mem_net)
        mem_net.eval()

        if self.workers.remote_workers():
            with self._timers[WORKER_UPDATE_TIMER]:
                self.workers.sync_weights(global_vars={"timestep": self._counters[NUM_AGENT_STEPS_SAMPLED]})

    def load_memory_weights(self, state_dict: dict):
        self.memup_module.memory_net.load_state_dict(state_dict)
        self.memup_module.memory_acc.load_state_dict(state_dict)
        mem_net = self.get_policy().model.lstm
        accumulate(self.memup_module.memory_net, mem_net, 0.0)

        if self.workers.remote_workers():
                self.workers.sync_weights()

    @override(Trainer)
    def setup(self, config: TrainerConfigDict):
        super().setup(config)
        logger.setLevel(logging.INFO)
        self.episodic_buffer = TrajectoryBuffer(config["memup"]["buffer_length"], DiscountedReturn(config["memup"]["gamma"]))
        self.memup_module = MemupModule(
            create_cuda_module(config["model"]["custom_model_config"]["module"]),
            create_cuda_module(config["memup"]["create_predictor"]),
            'return',
            torch.nn.MSELoss(reduction='none')
        )
        self.memup_train_op = MemupTrainOp(
            self.memup_module,
            self.episodic_buffer,
            config["memup"]["prediction_frequency"],
            config["memup"]["rollout_length"],
            torch.nn.MSELoss()
        )

        self.sync_memory()

        self.eval_buffer = TrajectoryBuffer(
            self.config["memup"]["eval_buffer_length"],
            Composite([DiscountedReturn(config["memup"]["gamma"]), AddTailEvalTargets(1)])
        )

        while (
                total_sampled_timesteps(self.workers.local_worker())
                < self.config["memup"]["prefill_timesteps"]
        ):
            samples = self.workers.local_worker().sample()
            self.add_samples_to_buffer(samples)

        self.eval_buffer.add_trajectories(self.episodic_buffer.get_data())
        self.eval_op = MemupEvalOp(self.memup_module, self.eval_buffer, [MSEMetric()])

    def add_samples_to_buffer(self, samples: SampleBatch):
        policy = self.get_policy()
        samples["obs_orig"] = restore_original_dimensions(
            samples["obs"], policy.observation_space, "numpy"
        )

        keys = samples["obs_orig"].keys()
        res = []
        episodes = samples.split_by_episode()
        # obs_shape = policy.observation_space.shape
        # num_actions = policy.action_space.n

        for e in episodes:
            if not e["dones"][-1]:
                continue
            data = {}
            for k in keys:
                data[k] = e["obs_orig"][k]
            data['action'] = e["actions"]
            data['reward'] = e["rewards"]
            data['done'] = e["dones"].astype(int)

            # data['observation_shape'] = obs_shape
            # data['num_actions'] = num_actions

            res.append(Trajectory(data))

        self.episodic_buffer.add_trajectories(res)
        if not self.eval_buffer.is_full:
            self.eval_buffer.add_trajectories(res)

        return samples

    @ExperimentalAPI
    def training_iteration(self) -> ResultDict:
        # Collect SampleBatches from sample workers until we have a full batch.
        logger.info(f"collect {self.config['train_batch_size']} samples")

        if self._by_agent_steps:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_agent_steps=self.config["train_batch_size"]
            )
        else:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config["train_batch_size"]
            )

        self.update_buffer_and_train_memup(train_batch)

        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["advantages"])
        # Train
        if self.config["simple_optimizer"]:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.remote_workers():
            with self._timers[WORKER_UPDATE_TIMER]:
                self.workers.sync_weights(global_vars=global_vars)

        # For each policy: update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            self.get_policy(policy_id).update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                self.config["vf_loss_coeff"] * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                log_once("ppo_warned_lr_ratio")
                and self.config.get("model", {}).get("vf_share_layers")
                and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                log_once("ppo_warned_vf_clip")
                and mean_reward > self.config["vf_clip_param"]
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)
        return train_results

    def update_buffer_and_train_memup(self, ppo_train_batch):
        train_every = self.config['memup_train_every']

        # If train_every is less than 1, this means that we do not train memup
        # If memup is not trained, then no need to fill the buffer
        if train_every <= 0: return

        train_batch = self.add_samples_to_buffer(ppo_train_batch)

        logger.info(f"episodes in buffer: {self.episodic_buffer.__len__()} / {self.episodic_buffer.maxlen}")
        logger.info(f"episodes in eval buffer: {self.eval_buffer.__len__()} / {self.eval_buffer.maxlen}")

        if (self._iteration + 1) % train_every == 0:
            logger.info("memup train")
            self.memup_train_op.exec(self.config["memup"]["batch_size"], self.config["memup"]["train_steps"])
            print()
            self.sync_memory()
            logger.info("policy train")


    @override(Trainer)
    def evaluate(
            self,
            episodes_left_fn=None,  # deprecated
            duration_fn: Optional[Callable[[int], int]] = None,
    ) -> dict:
        eval_info = super().evaluate(episodes_left_fn, duration_fn)
        #No need to test memory if it is not trained
        if self.config['memup_train_every'] >= 1:

            # print([traj.data["action"][-1] for traj in self.episodic_buffer.get_data()][-20:])
            memup_eval = self.eval_op.exec(self.config["memup"]["eval_buffer_length"])
            memup_eval_info = {"memory": {m.name: memup_eval[m.name] for m in self.eval_op.metrics}}
            eval_info = {**memup_eval_info, **eval_info}
        return eval_info

