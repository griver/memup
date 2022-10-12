from .common_wrappers import AddMemory
from .memory_len import make_pytorch_memory_len
from .common_wrappers import FrameStack
from .. import experiment_utils as exp_utils
import logging

try:
    from rlpyt.envs.gym import GymEnvWrapper as RLPytWrapper
except:
    RLPytWrapper = None
    logging.debug("can't find rlpyt package")


def make_pytorch_env(env_id, **kwargs):
    if env_id.lower().startswith("minigrid"):
        return make_pytorch_minigrid(env_id, **kwargs)
    elif env_id.lower().startswith("vizdoom"):
        return make_pytorch_vizdoom(env_id,**kwargs)
    elif env_id.lower().startswith('memory-len'):
        return make_pytorch_memory_len(env_id, **kwargs)
    else:
        return make_pytorch_atari(env_id, **kwargs)


def train_and_test_from_config(config):
    env_id = config['env_id']

    frst = config.get('framestack', 1)

    if env_id.startswith("MiniGrid"):
        from .env_minigrid import make_pytorch_minigrid
        train_env_args = config.setdefault(
            'train_env_args', dict(living_reward=-0.001, framestack=frst)
        )
        # default -0.001, #exploration=True
        env = make_pytorch_minigrid(env_id, **train_env_args)
        test_env = make_pytorch_minigrid(env_id, **train_env_args)

    elif env_id.lower().startswith('vizdoom'):
        from .env_vizdoom import make_pytorch_vizdoom
        train_env_args = config.setdefault(
            'train_env_args', dict(use_shaping=True, frame_skip=2, framestack=frst)
        )
        env = make_pytorch_vizdoom(env_id, id=1, **train_env_args)
        test_env = make_pytorch_vizdoom(env_id, id=7, **train_env_args)

    elif env_id.lower().startswith('memory-len') or \
            env_id.lower().startswith('t-maze'):
        train_env_args = config.setdefault('train_env_args', dict(framestack=frst))

        env = make_pytorch_memory_len(env_id, id=1, **train_env_args)
        test_env = make_pytorch_memory_len(env_id, id=7, **train_env_args)

    else:
        from .env_atari import make_pytorch_atari
        train_env_args = config.setdefault('train_env_args', dict(framestack=frst))
        env = make_pytorch_atari(env_id, **train_env_args)
        test_env = make_pytorch_atari(
            env_id, episode_life=False, clip_rewards=False, frame_stack=frst
        )

    #load external memory if needed
    if config.get('external_memory', False):
        memory_model = exp_utils.load_memory_model(config['memory_path'], env)
        env = AddMemory(env, memory_model)
        test_env = AddMemory(test_env, memory_model)
        config.setdefault('external_memory_dim', memory_model.lstm_size)
    else:
        config.setdefault('external_memory_dim', None)

    return env, test_env


def rlpyt_env_from_config(config):
    if not RLPytWrapper:
        raise ImportError("can't find rlpyt package")
    # we need to remove values but this dict is shared between envs:
    config = dict(config)
    memory_model = config.pop('memory_model', None)
    framestack = config.pop('framestack', 1)

    env = train_and_test_from_config(config)[0]

    if framestack > 1:
        env = FrameStack(env, framestack, greedy=True)

    if memory_model and not isinstance(env, AddMemory):
        env = AddMemory(env, memory_model)

    return RLPytWrapper(env)