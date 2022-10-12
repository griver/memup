import sys
import os
sys.path.append(os.getcwd())

import gym
from gym.wrappers import TimeLimit
from ray.tune import register_env
import ray
import torch
from rl.rllib.MemUpPPOTrainer import MemupPPOTrainer, RLPolicySummary
from examples.tmaze.rllib.policy import NoGradRNNModel
from examples.tmaze.tmaze_networks import TMazeRecMemory, TMazePredictor
from rl import envs
import yaml
import argparse
from memup.training import fix_seed


def load_ppo_config(args, create_mem, create_pred):

    with open(args.config) as f:
        ppo_config = yaml.load(f, Loader=yaml.SafeLoader)
    # this part is only relevant when policy and memory are trained in parallel:
    # should work for t-maze-1k, not tested for t-maze-20k
    ppo_config['memup'] = {
        "buffer_length": 2000,
        "eval_buffer_length": 10,
        "gamma": 0,
        "prefill_timesteps": 2000,
        "create_predictor": create_pred,
        "prediction_frequency": 1,
        "rollout_length": 10,
        "policy_accumulation_decay": 0.9,
        "batch_size": 128,
        "train_steps": 400,  # 400
    }
    # network to use with ppo
    # NoGrad means that RNN is learned by MemUP and takes no gradient updates from ppo
    ppo_config['model'] = {
        "custom_model": NoGradRNNModel,
        "max_seq_len": 10,
        "custom_model_config": {
            "module": create_mem,
        },
    }
    ppo_config['memup_train_every'] = args.update_frequency
    return ppo_config


def handle_commandline():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # episode generation arguments:
    parser.add_argument('-c', '--config', type=str, default=None, help='config for ppo')
    parser.add_argument('-l', '--length', type=int, default=1000, help='length of t-maze environment')
    parser.add_argument('-m', '--pretrained-memory', type=str, default=None, help='path to pretrained memory')
    parser.add_argument('-s', '--seed', type=int, default=1)
    parser.add_argument('-uf', '--update-frequency', type=int, default=1,
                        help='how often to update memup memory during ppo training.'
                             'update_frequency is set to -1, if pretrained memory is specified'
                        )

    parser.add_argument('-ld', '--logdir', type=str, default=None, help='path to save tensorboard logs')
    args = parser.parse_args()
    if args.logdir is None:
        args.logdir = f"logs/tmp/t-maze-{args.length}/ppo/seed{args.seed}/"

    if args.pretrained_memory is not None:
        args.update_frequency = -1

    return args

if __name__ == "__main__":
    args = handle_commandline()
    print(args)
    fix_seed(seed=args.seed)

    create_mem = lambda: TMazeRecMemory((3,), 2, 256)
    create_pred = lambda: TMazePredictor(256, (3,), 2, 'observation')
    config = load_ppo_config(args, create_mem, create_pred)

    ray.init()
    register_env("t-maze", lambda _:
            envs.train_and_test_from_config(dict(env_id=f't-maze-lnr-{args.length}'))[0])

    summary = RLPolicySummary(args.logdir)
    trainer = MemupPPOTrainer(env="t-maze", config=config)

    # default_config = {
    #     "gamma": 0.99,
    #     "num_gpus": 1,
    #     "num_workers": 8,
    #     "num_envs_per_worker": 4,
    #     "rollout_fragment_length": 2000, #2000,
    #     "sgd_minibatch_size": 5000, #1000,
    #     "train_batch_size": 100000,
    #     "batch_mode": "complete_episodes",
    #     "create_env_on_driver": True,
    #     "horizon": 20100,
    #     "entropy_coeff": 1e-2,
    #     "model": {
    #         "custom_model": NoGradRNNModel,
    #         "max_seq_len": 10,
    #         "custom_model_config": {
    #             "module": create_mem,
    #         },
    #     },
    #     'memup_train_every': args.update_frequency,
    #     "memup": {
    #         "buffer_length": 2000,
    #         "eval_buffer_length": 10,
    #         "gamma": 0,
    #         "prefill_timesteps": 2000,
    #         "create_predictor": create_pred,
    #         "prediction_frequency": 1,
    #         "rollout_length": 10,
    #         "policy_accumulation_decay": 0.9,
    #         "batch_size": 128,
    #         "train_steps": 400,  # 400
    #     },
    #     "framework": "torch",
    # }

    if args.pretrained_memory:
        weights = torch.load(args.pretrained_memory, map_location="cuda")["memory"]
        trainer.load_memory_weights(weights)

    try:
        for i in range(500):
            results = trainer.train()
            summary.add_training_step(results)

            print(f"Iter: {i}; avg. reward={results['episode_reward_mean']}")
            if i % 10 == 0:
                eval_results = trainer.evaluate()
                summary.add_eval_step(eval_results)
                print(
                    f"EVALUATION( policy_mean_reward = {eval_results['evaluation']['episode_reward_mean']} "
                    f"memory_loss = {eval_results['memory']['MSE']} )" if 'memory' in eval_results else ")"
                )

    finally:
        RLPolicySummary.close()


