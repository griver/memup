import os
import os.path as path
from os.path import join as join_path
import logging
import json
import yaml

import argparse
import torch as th
from gym import spaces

#========= loading or saving configs script and arguments =========
DEFAULT_CONFIG_DIR = "configs/"


def load_config(config_path, **replace_params):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    for k,v in replace_params.items():
        config[k] = v

    return config


def save_config(config, folder, file_name):
    file_path = join_path(folder, file_name)
    with open(file_path, 'w') as f:
        return yaml.dump(config, f, Dumper=yaml.SafeDumper)


def load_args(folder, file_name='main_args.json', default=None):
    file_path = join_path(folder, file_name)
    if os.path.isfile(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    return default


def save_args(args, folder, file_name='main_args.json', exclude_args=tuple()):
    args = vars(args) if isinstance(args, argparse.Namespace) else args
    save_args = {k:v for k,v in args.items() if k not in exclude_args}
    file_path = join_path(folder, file_name)
    ensure_dir(file_path)
    with open(file_path, 'w') as f:
        return json.dump(save_args, f,sort_keys=True,indent=2)


def get_train_config(model_path, parent_dir_lvl=2, **replace_params):

    assert model_path is not None or len(replace_params) > 0, \
        'at least one of the arguments should be not None'

    experiment_dir = get_root_dir(model_path, parent_dir_lvl)
    train_config = load_args(experiment_dir, default={})
    train_config.update(replace_params)

    return train_config


def ensure_dir(file_path):
    """
    Checks if the containing directories exist,
    and if not, creates them.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_root_dir(model_path, num_levels_up=2):
    folders = [path.dirname(model_path)]
    folders.extend(['..']*num_levels_up)
    return path.relpath(path.join(*folders))


def iterate_dirname(dirname, start_idx=1, max_idx=30):
    """
    If given directory already exists, adds index at the end of the name
    """
    if os.path.exists(dirname):
        base = dirname.rstrip('/')
        for i in range(start_idx, max_idx):
            dir = "{}_{}".format(base, i)
            if not os.path.exists(dir):
                break
        else:
            raise ValueError('folders from \n{}\n to \n{}\n already exist!'.format(base, dir))

        logging.info(
            'folder {} already exists, we created a folder {} instead'.format(base, dir))
        return dir

    return dirname

#======== /loading or saving configs script and arguments =========


#==================== loading pretrained models ===================
def load_qrdqn(model_path, obs_space, num_actions, device, train_config, verbose=False):
    if 'agent_config' in train_config:
        config = train_config['agent_config']
    else:

        default_path = join_path(DEFAULT_CONFIG_DIR, "distributional_rl/qrdqn.yaml")
        config_path = train_config.get('agent_config_path', default_path)

        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

    model = QRDQN(
        obs_space,
        num_actions,
        N=config['N'],
        dueling_net=config['dueling_net'],
        noisy_net=config['noisy_net'],
        external_memory_dim=train_config['external_memory_dim']
    ).to(device)

    checkpoint = th.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    if verbose:
        print_dict(config, 'QR-DQN Config')
        print(model)

    return model


def load_predictor_model(model_path, env, device='cpu', verbose=1):

    device = th.device(device)
    config = get_train_config(model_path, 0)
    if verbose:
        print_dict(config, 'Memory Model Config:')

    #stupid hack to deal with unknown number of
    # input frames for a memory model
    #here we just assume it's equal to 1
    n_frames = getattr(env, 'n_frames', 1)

    if isinstance(env.observation_space, spaces.Dict):

        logging.warning(
            "Memory model doesn't  expect dict"
            " obsevations {}".format(env.observation_space)
        )

        C, *extra_dims = env.observation_space['obs'].shape
    else:
        C, *extra_dims = env.observation_space.shape

    state_dict = th.load(model_path, map_location=device)
    C_mem = RecurrentMemory.num_channels_from_state_dict(state_dict['model_state'])
    obs_shape = (C_mem,) + tuple(extra_dims)

    predictor = create_memory_and_predictor(
        obs_shape,
        env.action_space.n,
        device,
        lstm_size=config['lstm_size'],
        lstm_layers=config['lstm_layers'],
        ignore_prev_reward=config['no_prev_reward']
    )
    # print("\nMODEL:",predictor, sep='\n')

    predictor.load_state_dict(state_dict['model_state'])
    predictor.eval()
    return predictor


def load_memory_model(model_path, env, device='cpu', verbose=1):
    predictor = load_predictor_model(model_path, env, device, verbose)
    return predictor.memory
#=================== /loading pretrained models ===================


#=========================== output funcs =========================
def red(line):
    return "\x1b[31;1m{0}\x1b[0m".format(line)


def yellow(line):
    return "\x1b[33;1m{0}\x1b[0m".format(line)


def green(line):
    return "\x1b[32;1m{0}\x1b[0m".format(line)


def blue(line):
    return "\x1b[34;1m{0}\x1b[0m".format(line)


def cyan(line):
    return "\x1b[36;1m{0}\x1b[0m".format(line)


def print_dict(d, name=None, offset=0, tab_len=4):
    if offset < 1:
        title = ' '.join(['==' * 10, '{}', '==' * 10])
        if name is not None:
            title = title.format(name)
        print(title)

    offset_pref = ' ' * (offset * tab_len)

    for k in sorted(d.keys()):
        if isinstance(d[k], dict) and len(d[k]) > 1:
            print(offset_pref, k, ":")
            print_dict(d[k], offset=(offset + 1), tab_len=tab_len)
        else:
            print(offset_pref, k, ':', d[k])

    if offset < 1:
        print('=' * len(title))
#==========================/ output funcs =========================
